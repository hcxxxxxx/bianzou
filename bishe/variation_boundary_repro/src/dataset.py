from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_mel(mel_t: np.ndarray) -> np.ndarray:
    mean = float(mel_t.mean())
    std = float(mel_t.std())
    if std < 1e-6:
        return mel_t - mean
    return (mel_t - mean) / std


def fold_sequence(seq: np.ndarray, fold_size: int) -> np.ndarray:
    n_folds = seq.shape[0] // fold_size
    if n_folds <= 0:
        return np.zeros((0,) + seq.shape[1:], dtype=seq.dtype)
    seq = seq[: n_folds * fold_size]
    return seq.reshape(n_folds, fold_size, *seq.shape[1:]).mean(axis=1)


def make_frame_labels(
    boundary_times: list[float],
    n_frames: int,
    sr: int,
    hop_length: int,
    tolerance: float,
) -> np.ndarray:
    labels = np.zeros(n_frames, dtype=np.float32)
    frame_duration = hop_length / sr
    duration = n_frames * frame_duration
    delta = max(0, int(round(tolerance / frame_duration)))
    for time_sec in boundary_times:
        if time_sec <= 0.0 or time_sec >= duration:
            continue
        center = int(round(time_sec / frame_duration))
        start = max(center - delta, 0)
        end = min(center + delta + 1, n_frames)
        labels[start:end] = 1.0
    return labels


class VariationBoundaryDataset(Dataset):
    def __init__(
        self,
        metadata: list[dict[str, Any]],
        stems: list[str],
        mel_dir: Path,
        sr: int = 44100,
        hop_length: int = 512,
        fold_seconds: float = 1.0,
        label_tolerance: float = 3.0,
    ) -> None:
        self.items = {item["filename"]: item for item in metadata}
        self.stems = list(stems)
        self.mel_dir = Path(mel_dir)
        self.sr = sr
        self.hop_length = hop_length
        self.fold_seconds = fold_seconds
        self.label_tolerance = label_tolerance
        self.fold_size = max(1, int(round(fold_seconds * sr / hop_length)))
        self.missing_features = [
            stem for stem in self.stems if not (self.mel_dir / f"{stem}.npy").exists()
        ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, Any]:
        stem = self.stems[index]
        item = self.items[stem]
        mel = np.load(self.mel_dir / f"{stem}.npy").astype(np.float32)
        mel_t = normalize_mel(mel.T)
        labels = make_frame_labels(
            item.get("boundary_times", []),
            n_frames=mel_t.shape[0],
            sr=self.sr,
            hop_length=self.hop_length,
            tolerance=self.label_tolerance,
        )
        folded_labels = fold_sequence(labels[:, None], self.fold_size).squeeze(-1)
        true_times = [
            float(t)
            for t in item.get("boundary_times", [])
            if t > 0.0 and t < mel_t.shape[0] * self.hop_length / self.sr
        ]
        return {
            "stem": stem,
            "mel": torch.from_numpy(mel_t),
            "labels": torch.from_numpy(folded_labels.astype(np.float32)),
            "true_times": true_times,
            "frame_length": mel_t.shape[0],
            "fold_length": folded_labels.shape[0],
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    n_mels = batch[0]["mel"].shape[1]
    max_frames = max(item["mel"].shape[0] for item in batch)
    max_folds = max(item["labels"].shape[0] for item in batch)

    mels = torch.zeros(batch_size, max_frames, n_mels, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_folds, dtype=torch.float32)
    label_mask = torch.zeros(batch_size, max_folds, dtype=torch.float32)
    frame_lengths = []
    fold_lengths = []
    stems = []
    true_times = []

    for i, item in enumerate(batch):
        frames = item["mel"].shape[0]
        folds = item["labels"].shape[0]
        mels[i, :frames] = item["mel"]
        labels[i, :folds] = item["labels"]
        label_mask[i, :folds] = 1.0
        frame_lengths.append(frames)
        fold_lengths.append(folds)
        stems.append(item["stem"])
        true_times.append(item["true_times"])

    return {
        "mels": mels,
        "labels": labels,
        "label_mask": label_mask,
        "frame_lengths": frame_lengths,
        "fold_lengths": fold_lengths,
        "stems": stems,
        "true_times": true_times,
    }

