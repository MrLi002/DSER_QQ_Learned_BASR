"""Compare Learned BASR V2 with repair enabled and disabled on matched frames.

Place this file next to interpolation.py. It uses the project's BS-ERGB
preprocessor and inference wrapper, and prints results to stdout.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model.checkpoints import configure_inference_args, read_checkpoint
from model.ourModel import Model
from predict import pred_utils
from predict.bsergb import indexing_skip_ind


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--frames-per-scene", type=int, default=3)
    parser.add_argument("--multi", type=int, default=1)
    parser.add_argument("--repair-ratio", type=float, default=None)
    args = parser.parse_args()
    if args.frames_per_scene < 1 or args.multi < 1:
        parser.error("--frames-per-scene and --multi must be positive")
    if args.repair_ratio is not None and not 0 < args.repair_ratio <= 1:
        parser.error("--repair-ratio must be in (0, 1] for the enabled run")
    return args


def sample_windows(n_images, scene, multi, limit):
    """Choose evenly spaced targets from the current evaluator's windows."""
    excluded = indexing_skip_ind.get(scene, ())
    windows = []
    for start in range(0, n_images - multi - 1, multi):
        end = start + multi + 1
        if any(start <= index < end for index in excluded):
            continue
        windows.extend((start, target, end) for target in range(start + 1, end))
    if len(windows) > limit:
        entries = np.linspace(0, len(windows) - 1, num=limit, dtype=int)
        windows = [windows[i] for i in entries]
    return windows


def measure_block(inputs, output):
    # The fourth positional argument is base_high, before the residual addition.
    repaired, aux = output
    difference = (repaired.float() - inputs[3].float()).abs()
    selected = aux["selected"].bool()
    valid = aux["valid0"].bool() | aux["valid1"].bool()
    gate = aux["gate"].float()
    external = aux["weights"][:, :2].float().sum(dim=1, keepdim=True)
    selected_count = selected.sum().clamp_min(1)
    values = (
        selected.float().mean(),
        (selected & valid).sum() / selected_count,
        (selected * gate).sum() / selected_count,
        (selected * gate * external).sum() / selected_count,
        difference.mean(),
        difference.amax(),
    )
    names = ("selected", "valid", "gate", "effective_usage", "write_mean", "write_max")
    return dict(zip(names, torch.stack(values).cpu().tolist()))


def rgb_and_metrics(prediction, gt):
    image = (prediction[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0)
    image = image.round().astype(np.uint8)
    return image, pred_utils.psnr_(gt, image), pred_utils.ssim_(gt, image)


@torch.inference_mode()
def main():
    args = parse_args()
    checkpoint = read_checkpoint(str(args.checkpoint))
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    args.basr_mode = "learned"
    args.beta = 0.1
    configure_inference_args(args, checkpoint)
    enabled_ratio = args.repair_ratio
    if not 0 < enabled_ratio <= 1:
        raise ValueError("Set an explicit positive --repair-ratio for the enabled run")

    wrapper = Model(args, inference_only=True)
    wrapper.load_checkpoint(str(args.checkpoint), checkpoint=checkpoint)
    net = wrapper.my_net
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    pred_utils.device = next(net.parameters()).device
    print(f"checkpoint={args.checkpoint} saved_epoch={checkpoint.get('epoch', 'unknown')} "
          f"saved_ratio={saved_args.get('repair_ratio', 'unknown')} "
          f"on_ratio={enabled_ratio:g} bins={args.bins} multi={args.multi} "
          f"device={pred_utils.device}", flush=True)

    stats = {}
    handles = []
    for name in ("refine_block0", "refine_block1"):
        block = getattr(net.sys, name)
        projection_abs_max = block.to_high.weight.float().abs().amax().item()
        print(f"{name} to_high_abs_max={projection_abs_max:.6e}")

        def hook(_block, inputs, output, name=name):
            stats[name] = measure_block(inputs, output)

        handles.append(block.register_forward_hook(hook))

    try:
        for scene in args.scenes:
            root = args.data_root / scene
            if not (root / "events").is_dir():
                raise FileNotFoundError(root / "events")
            images = sorted(str(path) for path in (root / "images").iterdir()
                            if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
            windows = sample_windows(len(images), scene, args.multi, args.frames_per_scene)
            if not windows:
                raise ValueError(f"No valid multi={args.multi} targets for {scene}")
            for start, target, end in windows:
                gt = cv2.imread(images[target], cv2.IMREAD_COLOR)
                if gt is None:
                    raise OSError(f"Could not read {images[target]}")
                height, width = gt.shape[:2]
                imgs = pred_utils.get_imgs(images, args.multi, start)
                voxels, event_mask = pred_utils.looking_for_event_index_by_timestamp(
                    str(root), args.bins, start, target, end, height, width, 240,
                    real=True, bsergb=True,
                )

                net.repair_ratio = 0.0
                stats.clear()
                off = wrapper.inference(imgs, voxels, event_mask).float()
                off_stats = stats.copy()
                net.repair_ratio = enabled_ratio
                stats.clear()
                on = wrapper.inference(imgs, voxels, event_mask).float()

                float_diff = (on - off).abs()
                rgb_off, psnr_off, ssim_off = rgb_and_metrics(off, gt)
                rgb_on, psnr_on, ssim_on = rgb_and_metrics(on, gt)
                changed = np.any(rgb_on != rgb_off, axis=2).mean()
                print(f"\n{scene} target={Path(images[target]).name} "
                      f"float_rgb_mean={float_diff.mean().item():.6e} "
                      f"float_rgb_max={float_diff.amax().item():.6e} "
                      f"uint8_changed_pixels={changed:.4%}", flush=True)
                print(f"  PSNR off={psnr_off:.6f} on={psnr_on:.6f} | "
                      f"SSIM off={ssim_off:.6f} on={ssim_on:.6f}")
                for name, values in stats.items():
                    off_values = off_stats[name]
                    print(f"  {name} off_selected={off_values['selected']:.4%} "
                          f"off_write_max={off_values['write_max']:.6e} "
                          f"on_selected={values['selected']:.4%} "
                          f"on_valid={values['valid']:.4%} on_gate={values['gate']:.6f} "
                          f"on_effective_usage={values['effective_usage']:.6f} "
                          f"on_write_mean={values['write_mean']:.6e} "
                          f"on_write_max={values['write_max']:.6e}", flush=True)
                if not stats:
                    raise RuntimeError("Neither Learned BASR V2 repair block ran")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
