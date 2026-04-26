from __future__ import annotations

import torch
import torch.nn.functional as F


def local_maxima(probs: torch.Tensor, filter_size: int = 9) -> torch.Tensor:
    if probs.ndim != 1:
        probs = probs.flatten()
    if probs.numel() == 0:
        return probs
    filter_size = max(1, int(filter_size))
    if filter_size % 2 == 0:
        filter_size += 1
    pad = filter_size // 2
    pooled = F.max_pool1d(probs.view(1, 1, -1), kernel_size=filter_size, stride=1, padding=pad)
    keep = probs == pooled.view(-1)
    return torch.where(keep, probs, torch.zeros_like(probs))


def process_prob_sections(
    probs: torch.Tensor,
    fold_seconds: float = 1.0,
    filter_size: int = 9,
    threshold: float = 0.05,
    max_predictions: int | None = None,
) -> list[float]:
    prob_sections = local_maxima(probs.detach().float().cpu(), filter_size=filter_size)
    pred_mask = prob_sections >= threshold
    indices = pred_mask.nonzero(as_tuple=False).flatten()
    if max_predictions is not None and max_predictions > 0 and indices.numel() > max_predictions:
        values = prob_sections[indices]
        order = torch.argsort(values, descending=True)[:max_predictions]
        indices = indices[order]
    indices = sorted(indices.tolist())
    return [round((idx + 0.5) * fold_seconds, 3) for idx in indices]
