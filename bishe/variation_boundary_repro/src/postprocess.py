from __future__ import annotations

import torch
import torch.nn.functional as F


def local_maxima(
    tensor: torch.Tensor,
    filter_size: int = 41,
    step: int = 1,
    mode: str = "author",
) -> tuple[torch.Tensor, list]:
    if mode == "maxpool":
        mode = "author"
    if mode not in {"author", "strict"}:
        raise ValueError(f"unknown peak mode: {mode}")

    if mode == "author":
        assert len(tensor.shape) in (1, 2), "Input tensor should have 1 or 2 dimensions"
        assert filter_size % 2 == 1, "Filter size should be an odd number"

        original_shape = tensor.shape
        if len(original_shape) == 1:
            tensor = tensor.unsqueeze(0)

        padding = filter_size // 2
        padded_arr = F.pad(tensor, (padding, padding), mode="constant", value=-torch.inf)
        rolling_view = padded_arr.unfold(1, filter_size, step)

        center = filter_size // 2
        local_maxima_mask = torch.eq(
            rolling_view[:, :, center],
            torch.max(rolling_view, dim=-1).values,
        )
        target_tensor = local_maxima_mask
        if step != 1:
            target_tensor = torch.zeros(
                (local_maxima_mask.shape[0], local_maxima_mask.shape[1] * step),
                dtype=torch.bool,
                device=tensor.device,
            )
            for i in range(local_maxima_mask.shape[1]):
                target_tensor[:, i * step] = local_maxima_mask[:, i]
            target_tensor = target_tensor[:, : tensor.shape[1]]

        output_arr = torch.zeros_like(tensor)
        output_arr[target_tensor] = tensor[target_tensor]
        output_arr = output_arr.reshape(original_shape)
        return output_arr, []

    probs = tensor
    if probs.ndim != 1:
        probs = probs.flatten()
    if probs.numel() == 0:
        return probs, []
    filter_size = max(1, int(filter_size))
    if filter_size % 2 == 0:
        filter_size += 1
    pad = filter_size // 2
    padded = F.pad(probs.view(1, 1, -1), (pad, pad), mode="constant", value=float("-inf"))
    windows = padded.unfold(dimension=2, size=filter_size, step=1).squeeze(0).squeeze(0)
    center = windows[:, pad]
    left_max = windows[:, :pad].max(dim=1).values if pad > 0 else torch.full_like(center, float("-inf"))
    right_max = windows[:, pad + 1 :].max(dim=1).values if pad > 0 else torch.full_like(center, float("-inf"))
    keep = (center > left_max) & (center > right_max)
    return torch.where(keep, probs, torch.zeros_like(probs)), []


def process_prob_sections(
    probs: torch.Tensor,
    fold_seconds: float = 1.0,
    filter_size: int = 9,
    step: int = 1,
    threshold: float = 0.0001,
    peak_mode: str = "author",
    time_position: str = "center",
    max_predictions: int | None = None,
    min_predictions: int = 0,
) -> list[float]:
    prob_sections, _ = local_maxima(
        probs.detach().float().cpu(),
        filter_size=filter_size,
        step=step,
        mode=peak_mode,
    )
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
