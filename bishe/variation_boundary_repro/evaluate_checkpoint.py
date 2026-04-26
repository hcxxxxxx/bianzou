#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import VariationBoundaryDataset, collate_batch
from src.metrics import evaluate_boundary_predictions
from src.model import SACNFolk
from src.postprocess import process_prob_sections
from src.splits import load_splits
from src.utils import default_data_root, load_metadata, save_json


def parse_float_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with post-processing sweeps.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=root / "artifacts" / "splits.json")
    parser.add_argument("--mel-dir", type=Path, default=root / "artifacts" / "mels")
    parser.add_argument("--checkpoint", type=Path, default=root / "artifacts" / "checkpoints" / "best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold-grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--filter-size-grid", type=str, default="5,7,9,11,15,21")
    parser.add_argument("--max-predictions-grid", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--oracle-count", action="store_true", help="Diagnostic only: keep top peaks equal to each song's true boundary count.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect_entries(model: SACNFolk, loader: DataLoader, device: torch.device) -> tuple[list[dict], float]:
    model.eval()
    entries = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        mels = batch["mels"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["label_mask"].to(device)
        logits = model(mels, frame_lengths=batch["frame_lengths"])
        labels = labels[:, : logits.shape[1]]
        mask = mask[:, : logits.shape[1]]
        loss = criterion(logits, labels) * mask
        total_loss += float(loss.sum().item())
        total_items += int(mask.sum().item())

        probs = torch.sigmoid(logits).cpu()
        for i, stem in enumerate(batch["stems"]):
            n = int(batch["fold_lengths"][i])
            entries.append(
                {
                    "filename": stem,
                    "probs": probs[i, :n],
                    "true_times": batch["true_times"][i],
                }
            )

    return entries, total_loss / max(total_items, 1)


def evaluate_entries(
    entries: list[dict],
    threshold: float,
    filter_size: int,
    fold_seconds: float,
    max_predictions: int,
    oracle_count: bool = False,
) -> dict:
    rows = []
    for entry in entries:
        cap = len(entry["true_times"]) if oracle_count else max_predictions
        pred_times = process_prob_sections(
            entry["probs"],
            fold_seconds=fold_seconds,
            filter_size=filter_size,
            threshold=threshold,
            max_predictions=cap if cap > 0 else None,
        )
        rows.append(
            {
                "filename": entry["filename"],
                "pred_times": pred_times,
                "true_times": entry["true_times"],
            }
        )
    metrics = evaluate_boundary_predictions(rows)
    return {
        "threshold": threshold,
        "filter_size": filter_size,
        "max_predictions_per_song": "oracle" if oracle_count else max_predictions,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    metadata_path = args.metadata or args.data_root.resolve() / "songs_dataset.json"
    metadata = load_metadata(metadata_path)
    splits = load_splits(args.splits)

    dataset = VariationBoundaryDataset(
        metadata=metadata,
        stems=splits["splits"][args.split],
        mel_dir=args.mel_dir,
        sr=ckpt_args["sr"],
        hop_length=ckpt_args["hop_length"],
        fold_seconds=ckpt_args["fold_seconds"],
        label_tolerance=ckpt_args["label_tolerance"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    model = SACNFolk(
        n_mels=ckpt_args["n_mels"],
        embed_dim=ckpt_args["embed_dim"],
        hidden_size=ckpt_args["hidden_size"],
        lstm_layers=ckpt_args["lstm_layers"],
        dropout=ckpt_args["dropout"],
        sr=ckpt_args["sr"],
        hop_length=ckpt_args["hop_length"],
        fold_seconds=ckpt_args["fold_seconds"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    entries, loss = collect_entries(model, loader, device)
    fold_seconds = model.fold_size * model.hop_length / model.sr
    results = []
    for filter_size in parse_int_grid(args.filter_size_grid):
        for threshold in parse_float_grid(args.threshold_grid):
            for max_predictions in parse_int_grid(args.max_predictions_grid):
                results.append(
                    evaluate_entries(
                        entries,
                        threshold=threshold,
                        filter_size=filter_size,
                        fold_seconds=fold_seconds,
                        max_predictions=max_predictions,
                    )
                )
            if args.oracle_count:
                results.append(
                    evaluate_entries(
                        entries,
                        threshold=threshold,
                        filter_size=filter_size,
                        fold_seconds=fold_seconds,
                        max_predictions=0,
                        oracle_count=True,
                    )
                )

    results.sort(key=lambda row: row["metrics"]["HR3"]["f1"], reverse=True)
    payload = {"split": args.split, "loss": loss, "best": results[:20]}
    for row in payload["best"][:10]:
        hr3 = row["metrics"]["HR3"]
        hr05 = row["metrics"]["HR.5"]
        print(
            f"HR3F={hr3['f1']:.4f} HR3P={hr3['precision']:.4f} HR3R={hr3['recall']:.4f} "
            f"HR.5F={hr05['f1']:.4f} pred={hr3['predicted']} "
            f"threshold={row['threshold']:.2f} filter={row['filter_size']} "
            f"max={row['max_predictions_per_song']}"
        )

    if args.output is not None:
        save_json(payload, args.output)
        print(f"Saved sweep to {args.output}")


if __name__ == "__main__":
    main()
