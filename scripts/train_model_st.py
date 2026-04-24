#!/usr/bin/env python3
"""Train model_st.AllInOneSectionOnly with batch_size=1 on songs_dataset.json.

This script assumes feature files are frame sequences saved as:
  <feature_dir>/<filename>.npy with shape (n_mels, T)
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model_st with batch_size=1.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--dataset-json", default="songs_dataset.json", help="Dataset JSON path")
    parser.add_argument("--feature-dir", default="features_mel81", help="Feature directory")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1, help="Must be 1 for this task")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fole-time", type=float, default=1.0)
    parser.add_argument("--dim-embed", type=int, default=81)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=3.0)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--run-dir", default="", help="Optional output run directory")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def install_natten_stub() -> None:
    """Install a minimal natten.functional stub so model_st can be imported."""
    natten_mod = types.ModuleType("natten")
    functional_mod = types.ModuleType("natten.functional")

    def _stub(*_: Any, **__: Any) -> None:
        raise RuntimeError("natten ops are not available in this environment")

    functional_mod.natten1dav = _stub
    functional_mod.natten1dqkrpb = _stub
    functional_mod.natten2dav = _stub
    functional_mod.natten2dqkrpb = _stub

    natten_mod.functional = functional_mod
    sys.modules["natten"] = natten_mod
    sys.modules["natten.functional"] = functional_mod


def import_model_st(root: Path):
    sys.path.insert(0, root.as_posix())
    try:
        return importlib.import_module("model_st")
    except ModuleNotFoundError as err:
        if err.name and err.name.startswith("natten"):
            install_natten_stub()
            return importlib.import_module("model_st")
        raise


class Head(nn.Module):
    """Simple linear head used by model_st.AllInOneSectionOnly."""

    def __init__(self, hs: int, num_classes: int = 1, init_confidence: float = 0.001):
        super().__init__()
        self.proj = nn.Linear(hs, num_classes)

        p = min(max(float(init_confidence), 1e-6), 1.0 - 1e-6)
        init_bias = np.log(p / (1.0 - p))
        nn.init.constant_(self.proj.bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeatureDataset(Dataset):
    """Load per-song features and build folded boundary labels."""

    def __init__(
        self,
        label_dict: dict[str, list[float]],
        feature_dir: str,
        sr: int = 44100,
        hop_length: int = 512,
        tolerance: float = 3.0,
        fole_time: float = 1.0,
    ):
        self.file_list = list(label_dict.keys())
        self.label_dict = label_dict
        self.feature_dir = feature_dir
        self.sr = sr
        self.hop_length = hop_length
        self.tolerance = tolerance
        self.fole_time = fole_time

    def __len__(self) -> int:
        return len(self.file_list)

    def generate_frame_labels(self, time_points: list[float], n_frames: int) -> np.ndarray:
        frame_labels = np.zeros(n_frames, dtype=np.float32)
        frame_duration = self.hop_length / self.sr
        delta = int(self.tolerance / frame_duration)

        for t in time_points:
            center = int(t / frame_duration)
            start = max(center - delta, 0)
            end = min(center + delta + 1, n_frames)
            frame_labels[start:end] = 1.0
        return frame_labels

    @staticmethod
    def fold_sequence(seq: np.ndarray, fold_size: int) -> np.ndarray:
        n = seq.shape[0]
        n_folds = n // fold_size
        seq = seq[: n_folds * fold_size]
        seq = seq.reshape(n_folds, fold_size, -1).mean(axis=1)
        return seq

    def __getitem__(self, idx: int):
        file_key = self.file_list[idx]
        file_id = file_key.replace(".wav", "")
        feature_path = Path(self.feature_dir) / f"{file_id}.npy"
        feat = np.load(feature_path)  # expected: (F, T)
        if feat.ndim != 2:
            raise ValueError(f"feature must be 2D, got {feat.ndim}D: {feature_path}")

        time_points = self.label_dict[file_key]
        n_frames = int(feat.shape[1])
        labels = self.generate_frame_labels(time_points, n_frames)

        # Convert feature to (T, F) for model input.
        feat = feat.T.astype(np.float32)
        frame_duration = self.hop_length / self.sr
        fold_frames = int(self.fole_time / frame_duration)
        folded_labels = self.fold_sequence(labels[:, None], fold_frames).squeeze()

        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(folded_labels, dtype=torch.float32),
            time_points,
        )


def load_records(dataset_json: Path) -> list[dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError("dataset json is empty or invalid")
    return records


def split_by_title(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Group-aware split with 8:1:1 ratio and zero title leakage.

    All versions of the same song title are assigned to one split only.
    The 8:1:1 target is applied on record counts (not title counts).
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[str(rec["title"])].append(rec)

    rng = random.Random(seed)
    grouped = list(groups.items())
    rng.shuffle(grouped)
    grouped.sort(key=lambda item: len(item[1]), reverse=True)

    total_records = len(records)
    targets = {
        "train": int(round(total_records * 0.8)),
        "val": int(round(total_records * 0.1)),
    }
    targets["test"] = total_records - targets["train"] - targets["val"]

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for _title, recs in grouped:
        size = len(recs)
        gaps = {k: targets[k] - split_counts[k] for k in ("train", "val", "test")}
        positive_gap_splits = [k for k in ("train", "val", "test") if gaps[k] > 0]

        if positive_gap_splits:
            # Prefer the split with the largest remaining gap.
            positive_gap_splits.sort(
                key=lambda k: (gaps[k], -split_counts[k] / max(targets[k], 1)),
                reverse=True,
            )
            chosen = positive_gap_splits[0]
        else:
            # All splits have reached targets; minimize post-assignment overflow.
            chosen = min(
                ("train", "val", "test"),
                key=lambda k: abs((split_counts[k] + size) - targets[k]),
            )

        splits[chosen].extend(recs)
        split_counts[chosen] += size

    return splits


def assert_no_title_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    title_sets = {
        name: {str(item["title"]) for item in items}
        for name, items in splits.items()
    }
    if title_sets["train"] & title_sets["val"]:
        raise RuntimeError("Title leakage detected between train and val")
    if title_sets["train"] & title_sets["test"]:
        raise RuntimeError("Title leakage detected between train and test")
    if title_sets["val"] & title_sets["test"]:
        raise RuntimeError("Title leakage detected between val and test")


def to_label_dict(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    label_dict: dict[str, list[float]] = {}
    for rec in records:
        key = f"{rec['filename']}.wav"
        label_dict[key] = [float(x) for x in rec["boundary_times"]]
    return label_dict


def validate_feature_shapes(records: list[dict[str, Any]], feature_dir: Path, dim_embed: int) -> None:
    missing: list[str] = []
    bad_shape: list[str] = []

    for rec in records:
        feature_path = feature_dir / f"{rec['filename']}.npy"
        if not feature_path.exists():
            missing.append(feature_path.as_posix())
            continue

        arr = np.load(feature_path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] != dim_embed:
            bad_shape.append(f"{feature_path.name}: shape={tuple(arr.shape)}")

    if missing:
        sample = "\n".join(missing[:8])
        raise FileNotFoundError(
            "Missing feature files. Please run extraction first. Sample missing files:\n" + sample
        )

    if bad_shape:
        sample = "\n".join(bad_shape[:8])
        raise ValueError(
            f"Feature shape mismatch: expected (dim_embed={dim_embed}, T). Sample bad files:\n" + sample
        )


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_accum_steps: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_steps = 0
    tp = fp = fn = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        chroma, labels, _time_points = batch
        x = chroma.squeeze(0).to(device)
        y = labels.squeeze(0).to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(x).reshape(-1)
            y = y.reshape(-1)

            n = min(logits.numel(), y.numel())
            logits = logits[:n]
            y = y[:n]

            loss = criterion(logits, y)

            if is_train:
                (loss / grad_accum_steps).backward()
                if step % grad_accum_steps == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        targets = (y > 0.0).float()

        tp += int(((preds == 1) & (targets == 1)).sum().item())
        fp += int(((preds == 1) & (targets == 0)).sum().item())
        fn += int(((preds == 0) & (targets == 1)).sum().item())

        total_loss += float(loss.item())
        n_steps += 1

    precision, recall, f1 = compute_prf(tp, fp, fn)
    return {
        "loss": total_loss / max(n_steps, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 for this task to avoid sequence distortion by padding/cropping")

    root = Path(args.root).resolve()
    dataset_json = (root / args.dataset_json).resolve()
    feature_dir = (root / args.feature_dir).resolve()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else (root / "runs" / f"model_st_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = pick_device(args.device)

    records = load_records(dataset_json)
    splits = split_by_title(records, seed=args.seed)
    assert_no_title_leakage(splits)

    validate_feature_shapes(records, feature_dir, dim_embed=args.dim_embed)

    model_st = import_model_st(root)
    model_st.Head = Head

    train_label_dict = to_label_dict(splits["train"])
    val_label_dict = to_label_dict(splits["val"])
    test_label_dict = to_label_dict(splits["test"])

    train_dataset = FeatureDataset(
        label_dict=train_label_dict,
        feature_dir=feature_dir.as_posix(),
        sr=44100,
        hop_length=512,
        tolerance=args.tolerance,
        fole_time=args.fole_time,
    )
    val_dataset = FeatureDataset(
        label_dict=val_label_dict,
        feature_dir=feature_dir.as_posix(),
        sr=44100,
        hop_length=512,
        tolerance=args.tolerance,
        fole_time=args.fole_time,
    )
    test_dataset = FeatureDataset(
        label_dict=test_label_dict,
        feature_dir=feature_dir.as_posix(),
        sr=44100,
        hop_length=512,
        tolerance=args.tolerance,
        fole_time=args.fole_time,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = model_st.AllInOneSectionOnly(
        fole_time=args.fole_time,
        dim_embed=args.dim_embed,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    split_summary = {
        "n_total": len(records),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_titles_total": len({r['title'] for r in records}),
        "n_titles_train": len({r['title'] for r in splits['train']}),
        "n_titles_val": len({r['title'] for r in splits['val']}),
        "n_titles_test": len({r['title'] for r in splits['test']}),
    }

    config_payload = {
        "args": vars(args),
        "device": str(device),
        "split_summary": split_summary,
    }
    save_json(run_dir / "config.json", config_payload)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []

    print("=== Training Start ===")
    print(f"device: {device}")
    print(f"run_dir: {run_dir}")
    print(f"split: train={split_summary['n_train']} val={split_summary['n_val']} test={split_summary['n_test']}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_accum_steps=max(args.grad_accum_steps, 1),
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                grad_accum_steps=1,
            )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        current_val = val_metrics["loss"]
        if current_val < best_val_loss - 1e-8:
            best_val_loss = current_val
            best_epoch = epoch
            no_improve = 0

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
                "split_summary": split_summary,
            }
            torch.save(ckpt, run_dir / "best.pt")
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={args.early_stop_patience})")
            break

    save_json(run_dir / "history.json", {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss})

    best_ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_accum_steps=1,
        )

    final_report = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
    }
    save_json(run_dir / "final_report.json", final_report)

    print("=== Training Done ===")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_loss: {best_val_loss:.6f}")
    print(
        "test_metrics: "
        f"loss={test_metrics['loss']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
