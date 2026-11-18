import torch
from torch import nn
from model.blocks import EventRecNet, FeaTNet, DenseBlock, frame_encoder, event_encoder, ref_encoder
from model.trans_sys import Transformer
from model.selective_basr import DifficultyHead, temporal_event_features, select_topk_patches
from util.utils_func import mask_reverse, pyramid_patch_mask


class MyNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        bins = args.bins
        self.beta = args.beta
        self.mask_patch_size = args.mask_patch_size
        self.basr_mode = getattr(args, "basr_mode", None) or "learned"
        self.basr_revision = 2
        if self.basr_mode not in ("original", "learned"):
            raise ValueError("basr_mode must be original or learned")
        self.repair_ratio = getattr(args, "repair_ratio", 0.25)
        self.use_temporal = not getattr(args, "disable_event_temporal", False)
        self.use_alignment = not getattr(args, "disable_alignment_candidates", False)
        self.use_modulation = not getattr(args, "disable_modulation_prior", False)
        self.event_rec = EventRecNet(2 * bins)
        self.TNet = FeaTNet(bins)
        self.fusion = DenseBlock(64, 3)
        unit_dim = 32
        self.scale = 3
        self.encoder_f = frame_encoder(3, unit_dim // 4)
        self.encoder_e = event_encoder(bins, unit_dim // 2)
        self.encoder_ref = ref_encoder(3, unit_dim // 2)
        self.sys = Transformer(
            unit_dim * 2, basr_mode=self.basr_mode,
            residual_radius=getattr(args, "repair_residual_radius", 2.0),
            chunk_size=getattr(args, "repair_chunk_size", 1024))
        if self.basr_mode == "learned":
            self.difficulty_head = DifficultyHead()

    def forward(self, imgs, voxels, mask, bins, return_aux=False, valid_mask=None):
        img0, img1 = imgs[:, :3], imgs[:, 6:9]
        v0t, vt1, v1t = voxels[:, :bins], voxels[:, bins:2 * bins], voxels[:, 2 * bins:3 * bins]
        pure_rec = self.event_rec(torch.cat((v0t, vt1), 1))
        rec = pure_rec * mask
        learned = self.basr_mode == "learned"
        if learned:
            F0, alignment0 = self.TNet(img0, img1, v0t, rec, return_alignment=True)
            F1, alignment1 = self.TNet(img1, img0, v1t, rec, return_alignment=True)
            offsets0, offsets1 = alignment0["offsets"], alignment1["offsets"]
        else:
            F0 = self.TNet(img0, img1, v0t, rec)
            F1 = self.TNet(img1, img0, v1t, rec)
        Ft = self.fusion(torch.cat((F0, F1), 1))

        f_frame0, f_frame1 = self.encoder_f(img0), self.encoder_f(img1)
        f_event0, f_event1 = self.encoder_e(v0t), self.encoder_e(vt1)
        frame_feature = [torch.cat((a, b), 1) for a, b in zip(f_frame0, f_frame1)]
        event_feature = [torch.cat((a, b), 1) for a, b in zip(f_event0, f_event1)]
        ref_feature = self.encoder_ref(Ft)
        aux = {}
        if learned:
            temporal = temporal_event_features(v0t, vt1)
            if not self.use_temporal:
                temporal = torch.zeros_like(temporal)
            difficulty = self.difficulty_head(img0, img1, Ft, temporal)
            selected, patch_scores = select_topk_patches(
                difficulty, self.mask_patch_size, self.repair_ratio, valid_mask)
            if not self.use_alignment:
                offsets0 = [torch.zeros_like(x) for x in offsets0]
                offsets1 = [torch.zeros_like(x) for x in offsets1]
            context = dict(offsets0=offsets0, offsets1=offsets1, temporal=temporal,
                           selected=selected, difficulty=difficulty, rgb0=img0, rgb1=img1,
                           valid_mask=valid_mask,
                           modulation0=alignment0["modulation"] if self.use_modulation and self.use_alignment else [None] * 3,
                           modulation1=alignment1["modulation"] if self.use_modulation and self.use_alignment else [None] * 3)
            predictions, stages = self.sys(event_feature, frame_feature, ref_feature, f_frame0, f_frame1, repair_context=context)
            corr_list = [torch.nn.functional.adaptive_max_pool2d(selected, f.shape[-2:]) for f in f_frame0]
            corr_show = selected
            aux = dict(difficulty=difficulty, patch_scores=patch_scores, selected=selected, stages=stages)
        else:
            corr0 = pyramid_patch_mask(img0, Ft, mask, self.mask_patch_size, self.beta)
            corr1 = pyramid_patch_mask(img1, Ft, mask, self.mask_patch_size, self.beta)
            for i in range(3):
                corr0[i], corr1[i] = mask_reverse(corr0[i], corr1[i])
            predictions = self.sys(event_feature, frame_feature, ref_feature, f_frame0, f_frame1, corr0, corr1)
            corr_list = [(a + b).clamp(max=1) for a, b in zip(corr0, corr1)]
            corr_show = corr_list[0]
        result = (pure_rec.repeat(1, 3, 1, 1), rec.repeat(1, 3, 1, 1), Ft, predictions, corr_show, corr_list)
        return result + (aux,) if return_aux else result

