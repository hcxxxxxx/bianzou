from __future__ import annotations

import torch
import torch.nn.functional as F


def local_maxima(probs: torch.Tensor, filter_size: int = 9, mode: str = "maxpool") -> torch.Tensor:
    if probs.ndim != 1:
        probs = probs.flatten()
    if probs.numel() == 0:
        return probs
    filter_size = max(1, int(filter_size))
    if filter_size % 2 == 0:
        filter_size += 1
    pad = filter_size // 2
    if mode == "maxpool":
        pooled = F.max_pool1d(probs.view(1, 1, -1), kernel_size=filter_size, stride=1, padding=pad)
        keep = probs == pooled.view(-1)
    elif mode == "strict":
        padded = F.pad(probs.view(1, 1, -1), (pad, pad), mode="constant", value=float("-inf"))
        windows = padded.unfold(dimension=2, size=filter_size, step=1).squeeze(0).squeeze(0)
        center = windows[:, pad]
        left_max = windows[:, :pad].max(dim=1).values if pad > 0 else torch.full_like(center, float("-inf"))
        right_max = windows[:, pad + 1 :].max(dim=1).values if pad > 0 else torch.full_like(center, float("-inf"))
        keep = (center > left_max) & (center > right_max)
    else:
        raise ValueError(f"unknown peak mode: {mode}")
    return torch.where(keep, probs, torch.zeros_like(probs))


def process_prob_sections(
    probs: torch.Tensor,
    fold_seconds: float = 1.0,
    filter_size: int = 9,
    threshold: float = 0.0001,
    peak_mode: str = "maxpool",
    time_position: str = "center",
    max_predictions: int | None = None,
    min_predictions: int = 0,
) -> list[float]:
    prob_sections = local_maxima(probs.detach().float().cpu(), filter_size=filter_size, mode=peak_mode)
    pred_mask = prob_sections >= threshold
    indices = pred_mask.nonzero(as_tuple=False).flatten()
    positive_peaks = (prob_sections > 0).nonzero(as_tuple=False).flatten()
    if min_predictions > 0 and indices.numel() < min_predictions and positive_peaks.numel() > indices.numel():
        keep_count = min(min_predictions, positive_peaks.numel())
        values = prob_sections[positive_peaks]
        order = torch.argsort(values, descending=True)[:keep_count]
        indices = positive_peaks[order]
    if max_predictions is not None and max_predictions > 0 and indices.numel() > max_predictions:
        values = prob_sections[indices]
        order = torch.argsort(values, descending=True)[:max_predictions]
        indices = indices[order]
    if time_position == "center":
        offset = 0.5
    elif time_position == "start":
        offset = 0.0
    else:
        raise ValueError(f"unknown time position: {time_position}")
    indices = sorted(indices.tolist())
    return [round((idx + offset) * fold_seconds, 3) for idx in indices]
