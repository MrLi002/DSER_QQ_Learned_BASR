"""Strict backbone loading, with an explicit original-to-learned BASR migration."""
from __future__ import annotations
import torch


def read_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    state = checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in ("net", "model", "state_dict"):
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    normalized = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("module."):
            key = key[7:]
        key = key.replace(".transformer_block.to_patch_embedding2.0.",
                          ".transformer_block.to_patch_embedding.1.")
        normalized[key] = value
    if not normalized:
        raise ValueError("No model tensors found in checkpoint")
    return normalized


def checkpoint_mode(checkpoint):
    return "learned" if any(k.startswith("difficulty_head.")
                            for k in extract_state_dict(checkpoint)) else "original"


def checkpoint_revision(checkpoint):
    state = extract_state_dict(checkpoint)
    if not any(key.startswith("difficulty_head.") for key in state):
        return 0
    return 2 if any(".retriever.modulation_projection." in key for key in state) else 1


def load_network_state(model, checkpoint, allow_basr_migration=False, allow_partial=False):
    network = model.module if isinstance(model, torch.nn.DataParallel) else model
    state = extract_state_dict(checkpoint)
    source_mode = checkpoint_mode(checkpoint)
    target_mode = getattr(network, "basr_mode", source_mode)
    source_revision = checkpoint_revision(checkpoint)
    target_revision = getattr(network, "basr_revision", source_revision)
    changed = source_mode != target_mode or (source_mode == "learned" and source_revision != target_revision)
    migration = (changed and target_mode == "learned" and allow_basr_migration
                 and source_revision < target_revision)
    repair_prefixes = ("sys.refine_block0.", "sys.refine_block1.")
    dropped = []
    if migration:
        dropped = [key for key in state if key.startswith(repair_prefixes)]
        state = {key: value for key, value in state.items() if key not in dropped}
    elif changed:
        raise RuntimeError("Checkpoint/model BASR modes differ. Use --pretrained for "
                           "old-to-new BASR migration, or select the matching code version.")
    incompatible = network.load_state_dict(state, strict=False)
    missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
    allowed = (repair_prefixes + (("difficulty_head.",) if source_mode == "original" else ())) if migration else ()
    forbidden_missing = [key for key in missing if not key.startswith(allowed)]
    if not allow_partial and (forbidden_missing or unexpected):
        raise RuntimeError(f"Backbone/checkpoint mismatch: missing={forbidden_missing}; "
                           f"unexpected={unexpected}")
    return dict(migrated=migration, missing=missing, unexpected=unexpected, dropped=dropped,
                source_revision=source_revision, target_revision=target_revision)


MODEL_DEFAULTS = dict(bins=8, mask_patch_size=32, repair_ratio=0.25,
                      repair_residual_radius=2.0, repair_chunk_size=1024,
                      disable_event_temporal=False, disable_alignment_candidates=False,
                      disable_modulation_prior=False)


def configure_inference_args(args, checkpoint):
    if checkpoint_mode(checkpoint) == "learned" and checkpoint_revision(checkpoint) != 2:
        raise ValueError("This is a V1 learned-BASR checkpoint. Use the V1 code for inference, "
                         "or --pretrained with V2 training to migrate the backbone.")
    saved = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(saved, dict):
        saved = vars(saved)
    for key, default in MODEL_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, saved.get(key, default))
    if getattr(args, "basr_mode", None) is None:
        args.basr_mode = checkpoint_mode(checkpoint)
    return args


def validate_resume_args(args, checkpoint):
    saved = checkpoint.get("args", {})
    expected_mode = checkpoint_mode(checkpoint)
    if args.basr_mode != expected_mode:
        raise ValueError("--resume requires the same architecture. To start learned BASR "
                         "from original weights, use --pretrained instead.")
    if expected_mode == "learned" and checkpoint_revision(checkpoint) != 2:
        raise ValueError("V1-to-V2 migration requires --pretrained, not --resume")
    keys = ("bins", "mask_patch_size", "repair_ratio", "repair_residual_radius",
            "disable_event_temporal", "disable_alignment_candidates", "disable_modulation_prior")
    for key in keys:
        if key in saved and getattr(args, key) != saved[key]:
            raise ValueError(f"Resume configuration differs for {key}: "
                             f"current={getattr(args, key)}, saved={saved[key]}")
