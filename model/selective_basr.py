"""Learned BASR: error selection, target-grid support retrieval and gated repair.

No ground truth is accepted by any forward method. GT is used only by
``basr_auxiliary_losses``. Offsets use torchvision's (dy, dx) channel order.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def temporal_event_features(left, right):
    """Seven activity/time features; |voxel| is a proxy, not raw event counts.

    Each segment is already independently time-normalized by the data loader.
    These features therefore do not estimate physical event rates or occlusion
    labels. Signed voxel cancellation is another reason to treat them as cues.
    """
    def stats(voxel):
        magnitude = voxel.float().abs()
        mass = magnitude.sum(1, keepdim=True)
        time = torch.linspace(0, 1, voxel.shape[1], device=voxel.device)
        time = time.view(1, -1, 1, 1)
        mean = (magnitude * time).sum(1, keepdim=True) / mass.clamp_min(1e-6)
        variance = (magnitude * (time - mean).square()).sum(1, keepdim=True)
        variance = variance / mass.clamp_min(1e-6)
        activity = torch.log1p(mass)
        activity = activity / (1 + activity.mean((-2, -1), keepdim=True))
        return torch.cat((activity, mean, variance), 1), mass
    a, mass0 = stats(left)
    b, mass1 = stats(right)
    balance = (mass0 - mass1) / (mass0 + mass1).clamp_min(1e-6)
    return torch.cat((a, b, balance), 1).to(left.dtype)


class DifficultyHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, hidden, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.constant_(self.net[-1].bias, math.log(0.05 / 0.95))

    def forward(self, image0, image1, coarse, temporal):
        # Isolate the error-head objective from the reconstruction backbone.
        inputs = torch.cat((image0, image1, coarse.detach(), temporal), 1)
        return self.net(inputs).float().sigmoid()


def select_topk_patches(error, patch_size=32, ratio=0.25, valid_mask=None):
    """Select ceil(ratio * valid_patch_count) patches separately per sample.

    Hard ranking is intentionally non-differentiable. The error head has a
    direct supervised objective, instead of relying on gradients through top-k.
    """
    if patch_size <= 0 or not 0 <= ratio <= 1:
        raise ValueError("patch_size must be positive and ratio in [0, 1]")
    if error.ndim != 4 or error.shape[1] != 1:
        raise ValueError("error must have shape [B, 1, H, W]")
    valid = torch.ones_like(error) if valid_mask is None else valid_mask.to(error)
    if valid.shape != error.shape:
        raise ValueError("valid_mask and error must have identical shapes")
    p = patch_size
    numerator = F.avg_pool2d(error.detach() * valid, p, p, ceil_mode=True,
                             count_include_pad=False)
    denominator = F.avg_pool2d(valid, p, p, ceil_mode=True, count_include_pad=False)
    scores = numerator / denominator.clamp_min(1e-6)
    active = denominator > 0
    selected = torch.zeros_like(scores)
    for batch in range(error.shape[0]):
        candidates = active[batch].flatten().nonzero(as_tuple=False).flatten()
        k = min(candidates.numel(), math.ceil(ratio * candidates.numel()))
        if k:
            values = scores[batch].flatten()[candidates]
            indices = candidates[torch.topk(values, k, sorted=False).indices]
            selected[batch].view(-1)[indices] = 1
    mask = selected.repeat_interleave(p, -2).repeat_interleave(p, -1)
    return mask[..., :error.shape[-2], :error.shape[-1]] * valid, scores


class TargetGridRetriever(nn.Module):
    """Two-pass retrieval: sample coarse candidates, then predict residuals.

    The residual predictor sees source features already transported onto the
    target grid. DCN offsets/modulation inputs are detached; their original
    stage-B optimization remains unchanged.
    """
    def __init__(self, channels, residual_radius=2.0, chunk_size=1024):
        super().__init__()
        if residual_radius < 0 or chunk_size < 1:
            raise ValueError("Invalid retrieval radius or chunk size")
        self.radius = float(residual_radius)
        self.chunk_size = int(chunk_size)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        # target C + coarse-aligned raw source C + time cues 7 + validity 1
        self.offset_residual = nn.Sequential(
            nn.Conv2d(channels * 2 + 8, channels, 3, padding=1),
            nn.LeakyReLU(0.1), nn.Conv2d(channels, 18, 1))
        nn.init.zeros_(self.offset_residual[-1].weight)
        nn.init.zeros_(self.offset_residual[-1].bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))
        # Original DCN modulation is signed/unbounded. Map it to a learned
        # positive distribution instead of taking log(raw_m).
        self.modulation_projection = nn.Conv2d(9, 9, 1)
        with torch.no_grad():
            self.modulation_projection.weight.copy_(torch.eye(9).view(9, 9, 1, 1))
            self.modulation_projection.bias.zero_()
        self.prior_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        kernel = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)]
        self.register_buffer("kernel_yx", torch.tensor(kernel).float(), persistent=False)

    def _modulation_prior(self, modulation, reference):
        b, _, h, w = reference.shape
        if modulation is None:
            return reference.new_zeros(b, 9, h * w).float()
        if modulation.shape != (b, 9, h, w):
            raise ValueError("DCN modulation must have shape [B, 9, H, W]")
        mapped = self.modulation_projection(modulation.detach().to(reference)).float()
        return (self.prior_logit.sigmoid() * mapped.log_softmax(1)).flatten(2)

    def _sample(self, query_map, maps, offsets, selected, prior):
        b, c, h, w = query_map.shape
        all_offsets = offsets.reshape(b, 9, 2, h * w).float()
        temperature = self.log_temperature.exp().clamp(0.03, 1.0)
        batches = [[] for _ in range(5)]
        for batch in range(b):
            indices = (selected[batch, 0].flatten() > 0).nonzero(as_tuple=False).flatten()
            results = [query_map.new_zeros(channels, h * w)
                       for channels in (c, 3, 1, 1, c)]
            for start in range(0, indices.numel(), self.chunk_size):
                idx = indices[start:start + self.chunk_size]
                delta = all_offsets[batch, :, :, idx].permute(2, 0, 1)
                yx = torch.stack((idx // w, idx % w), -1).float()[:, None, :]
                sample_yx = yx + self.kernel_yx[None] + delta
                sy, sx = sample_yx[..., 0], sample_yx[..., 1]
                valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
                grid = torch.stack((2 * sx / max(w - 1, 1) - 1,
                                    2 * sy / max(h - 1, 1) - 1), -1)
                sampled = F.grid_sample(maps[batch:batch + 1], grid[None],
                                        align_corners=True, padding_mode="zeros")[0]
                keys, values, rgb, sampled_valid, raw_source = sampled.split((c, c, 3, 1, c), 0)
                valid = valid & (sampled_valid[0] >= 0.999)
                query = query_map[batch].reshape(c, -1)[:, idx].float()
                similarity = (F.normalize(query, dim=0)[:, :, None]
                              * F.normalize(keys, dim=0)).sum(0)
                logits = similarity / temperature + prior[batch, :, idx].transpose(0, 1)
                logits = logits.masked_fill(~valid, -1e4)
                attention = logits.softmax(-1) * valid.float()
                attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
                updates = ((values * attention[None]).sum(-1),
                           (rgb * attention[None]).sum(-1),
                           valid.any(-1).float()[None],
                           (similarity * attention).sum(-1)[None],
                           (raw_source * attention[None]).sum(-1))
                results = [value.index_copy(1, idx, update.to(query_map.dtype))
                           for value, update in zip(results, updates)]
            for bucket, value in zip(batches, results):
                bucket.append(value.view(-1, h, w))
        return tuple(torch.stack(bucket) for bucket in batches)

    def forward(self, target, source, rgb, offsets, temporal, selected,
                source_valid=None, modulation=None):
        b, c, h, w = target.shape
        if source.shape != target.shape or offsets.shape != (b, 18, h, w):
            raise ValueError("E-PCD offsets and features must share the target scale")
        if rgb.shape != (b, 3, h, w):
            raise ValueError("Source RGB must match the feature scale")
        source_valid = torch.ones_like(source[:, :1]) if source_valid is None else source_valid.to(source)
        query = self.query(target)
        maps = torch.cat((self.key(source), self.value(source), rgb.to(source),
                          source_valid, source), 1).float()
        prior = self._modulation_prior(modulation, target)
        coarse_offsets = offsets.detach()
        coarse = self._sample(query, maps, coarse_offsets, selected, prior)
        coarse_aligned, coarse_valid = coarse[4], coarse[2]
        residual = self.radius * torch.tanh(self.offset_residual(torch.cat(
            (target, coarse_aligned, temporal.to(target), coarse_valid), 1)))
        refined = self._sample(query, maps, coarse_offsets + residual, selected, prior)
        return refined[:4]


class LearnedDetailsSup(nn.Module):
    def __init__(self, channels, high_channels, residual_radius=2.0, chunk_size=1024):
        super().__init__()
        self.retriever = TargetGridRetriever(channels, residual_radius, chunk_size)
        # Latents 3C; time/error/validity/matching 12; explicit RGB evidence 15.
        # Remove uncalibrated |decoder_feature - value(source_feature)| terms.
        self.decision = nn.Sequential(
            nn.Conv2d(channels * 3 + 27, channels * 2, 3, padding=1),
            nn.LeakyReLU(0.1), nn.Conv2d(channels * 2, 4, 1))
        nn.init.zeros_(self.decision[-1].weight)
        with torch.no_grad():
            self.decision[-1].bias.copy_(torch.tensor([0., 0., 2., -2.]))
        self.to_high = nn.Conv2d(channels, high_channels, 1, bias=False)
        nn.init.zeros_(self.to_high.weight)

    def forward(self, target, source0, source1, base_high, offsets0, offsets1,
                temporal, selected, difficulty, rgb0, rgb1, base_rgb, valid_mask=None,
                modulation0=None, modulation1=None):
        size = target.shape[-2:]
        temporal = F.interpolate(temporal, size=size, mode="bilinear", align_corners=False)
        selected_low = F.adaptive_max_pool2d(selected, size).to(target)
        error_low = F.interpolate(difficulty.detach(), size=size, mode="area").to(target)
        rgb0 = F.interpolate(rgb0, size=size, mode="bilinear", align_corners=False)
        rgb1 = F.interpolate(rgb1, size=size, mode="bilinear", align_corners=False)
        source_valid = None if valid_mask is None else F.interpolate(valid_mask, size=size, mode="nearest")
        support0, color0, valid0, score0 = self.retriever(
            target, source0, rgb0, offsets0, temporal, selected_low, source_valid, modulation0)
        support1, color1, valid1, score1 = self.retriever(
            target, source1, rgb1, offsets1, temporal, selected_low, source_valid, modulation1)
        base_rgb = base_rgb.to(target)
        inputs = torch.cat((target, support0, support1, temporal.to(target), error_low,
                            valid0, valid1, score0, score1, color0, color1, base_rgb,
                            (color0 - base_rgb).abs(), (color1 - base_rgb).abs()), 1)
        decision = self.decision(inputs).float()
        logits = torch.cat((decision[:, :1].masked_fill(valid0 == 0, -1e4),
                            decision[:, 1:2].masked_fill(valid1 == 0, -1e4),
                            decision[:, 2:3]), 1)
        weights = logits.softmax(1).to(target)
        gate_logits = decision[:, 3:4]
        gate = gate_logits.sigmoid().to(target)
        fused = (weights[:, :1] * support0 + weights[:, 1:2] * support1
                 + weights[:, 2:3] * target)
        delta = selected_low * gate * (fused - target)
        delta = F.interpolate(delta, size=base_high.shape[-2:], mode="bilinear",
                              align_corners=False)
        repair_high = F.adaptive_max_pool2d(selected, base_high.shape[-2:]).to(base_high)
        output = base_high + repair_high * self.to_high(delta)
        aux = dict(route_logits=logits, gate_logits=gate_logits, weights=weights,
                   gate=gate, selected=selected_low, rgb0=color0, rgb1=color1,
                   base_rgb=base_rgb, valid0=valid0, valid1=valid1)
        return output, aux


def basr_auxiliary_losses(aux, coarse, gt, patch_size=32, temperature=0.05,
                          gate_margin=0.02):
    """Detached route/gate labels plus an independent differentiable RGB loss.

    Retrieval loss is not multiplied by gate, source weights or to_high. Its
    best-valid-side Charbonnier objective trains sampling/matching from step 1.
    Validity is a sampling condition, not a ground-truth visibility label.
    """
    predicted_error = aux["difficulty"].float()
    error_target = (coarse.detach().float() - gt.float()).abs().mean(1, keepdim=True)
    difficulty = F.smooth_l1_loss(predicted_error, error_target, beta=0.01)
    difficulty = difficulty + F.l1_loss(
        F.avg_pool2d(predicted_error, patch_size, patch_size, ceil_mode=True),
        F.avg_pool2d(error_target, patch_size, patch_size, ceil_mode=True))
    losses = {name: [] for name in ("route", "gate", "retrieve")}
    metrics = {name: [] for name in ("source_usage", "gate_mean", "effective_usage",
                                   "support_valid", "retrieval_error", "base_error",
                                   "oracle_improvable")}
    for stage in aux["stages"]:
        selected = stage["selected"].float()
        target = F.interpolate(gt.float(), size=selected.shape[-2:], mode="bilinear",
                               align_corners=False)
        colors = (stage["rgb0"].float(), stage["rgb1"].float())
        valid_sources = torch.cat((stage["valid0"], stage["valid1"]), 1).bool()
        eligible = selected * valid_sources.any(1, keepdim=True).float()
        eligible_count = eligible.sum().clamp_min(1)
        # This computation must remain outside no_grad and must not detach RGB.
        rgb_errors = torch.cat([(torch.sqrt((rgb - target).square() + 1e-6)
                                 - 1e-3).mean(1, keepdim=True) for rgb in colors], 1)
        best_rgb_error = rgb_errors.masked_fill(~valid_sources, 1e4).amin(1, keepdim=True)
        # Invalid/all-empty samples contribute zero while preserving the graph.
        losses["retrieve"].append((best_rgb_error * eligible).sum() / eligible_count)
        with torch.no_grad():
            errors = torch.cat([(rgb.detach().float() - target).abs().mean(1, keepdim=True)
                                for rgb in (*colors, stage["base_rgb"])], 1)
            valid = torch.cat((valid_sources, torch.ones_like(valid_sources[:, :1])), 1)
            targets = (-errors / temperature).masked_fill(~valid, -1e4).softmax(1)
            best_support = errors[:, :2].masked_fill(~valid_sources, 1e4).amin(1, keepdim=True)
            gate_target = ((errors[:, 2:3] - best_support) / gate_margin).clamp(0, 1)
            def average(value):
                return (value * eligible).sum() / eligible_count
            external = stage["weights"][:, :2].float().sum(1, keepdim=True)
            gate = stage["gate"].float()
            metrics["source_usage"].append(average(external))
            metrics["gate_mean"].append(average(gate))
            metrics["effective_usage"].append(average(external * gate))
            metrics["support_valid"].append(eligible.sum() / selected.sum().clamp_min(1))
            metrics["retrieval_error"].append(average(best_support))
            metrics["base_error"].append(average(errors[:, 2:3]))
            metrics["oracle_improvable"].append(average((best_support < errors[:, 2:3]).float()))
        denominator = selected.sum().clamp_min(1)
        route_map = -(targets * stage["route_logits"].log_softmax(1)).sum(1, keepdim=True)
        gate_map = F.binary_cross_entropy_with_logits(stage["gate_logits"], gate_target,
                                                     reduction="none")
        losses["route"].append((route_map * selected).sum() / denominator)
        losses["gate"].append((gate_map * selected).sum() / denominator)
    zero = predicted_error.sum() * 0
    result = {"difficulty": difficulty}
    result.update({name: torch.stack(values).mean() if values else zero
                   for name, values in losses.items()})
    result.update({name: torch.stack(values).mean() if values else zero.detach()
                   for name, values in metrics.items()})
    return result
