#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VariationBoundaryDataset, collate_batch
from src.metrics import evaluate_boundary_predictions
from src.model import SACNFolk
from src.postprocess import process_prob_sections
from src.splits import create_and_save_splits, load_splits
from src.utils import default_data_root, ensure_dir, load_metadata, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train SA-CNFolk reproduction.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=root / "artifacts" / "splits.json")
    parser.add_argument("--mel-dir", type=Path, default=root / "artifacts" / "mels")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "artifacts" / "checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--fold-seconds", type=float, default=1.0)
    parser.add_argument("--label-tolerance", type=float, default=3.0)
    parser.add_argument(
        "--no-normalize-mel",
        dest="normalize_mel",
        action="store_false",
        help="Use raw log-Mel dB values, matching the author's feature-saving snippet.",
    )
    parser.set_defaults(normalize_mel=True)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--filter-size", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="0.0001,0.001,0.005,0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80",
        help="Comma-separated thresholds searched on validation data.",
    )
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument(
        "--max-predictions-per-song",
        type=int,
        default=0,
        help="Keep only the strongest N local maxima per song; 0 keeps all peaks.",
    )
    parser.add_argument(
        "--min-predictions-per-song",
        type=int,
        default=0,
        help="Keep at least N strongest local maxima per song after thresholding; 0 disables.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--selection-metric",
        choices=["HR3", "SEG3"],
        default="HR3",
        help="Metric used for threshold tuning, scheduler, and checkpoint selection.",
    )
    return parser.parse_args()


def serialize_args(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def make_loader(dataset: VariationBoundaryDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    loss = criterion(logits, labels)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def train_one_epoch(
    model: SACNFolk,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accum_steps: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_items = 0

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        mels = batch["mels"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["label_mask"].to(device)
        logits = model(mels, frame_lengths=batch["frame_lengths"])
        labels = labels[:, : logits.shape[1]]
        mask = mask[:, : logits.shape[1]]
        loss = masked_bce_loss(logits, labels, mask, criterion)
        (loss / accum_steps).backward()

        if step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_items = int(mask.sum().item())
        total_loss += float(loss.item()) * batch_items
        total_items += batch_items

    if len(loader) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_items, 1)


def rows_from_prob_entries(
    entries: list[dict],
    threshold: float,
    fold_seconds: float,
    filter_size: int,
    max_predictions_per_song: int,
    min_predictions_per_song: int,
) -> list[dict]:
    max_predictions = max_predictions_per_song if max_predictions_per_song > 0 else None
    rows = []
    for entry in entries:
        pred_times = process_prob_sections(
            entry["probs"],
            fold_seconds=fold_seconds,
            filter_size=filter_size,
            threshold=threshold,
            max_predictions=max_predictions,
            min_predictions=min_predictions_per_song,
        )
        rows.append(
            {
                "filename": entry["filename"],
                "pred_times": pred_times,
                "true_times": entry["true_times"],
            }
        )
    return rows


@torch.no_grad()
def evaluate(
    model: SACNFolk,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, dict, list[dict]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    entries: list[dict] = []
    fold_seconds = model.fold_size * args.hop_length / args.sr

    for batch in tqdm(loader, desc="eval", leave=False):
        mels = batch["mels"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["label_mask"].to(device)
        logits = model(mels, frame_lengths=batch["frame_lengths"])
        labels = labels[:, : logits.shape[1]]
        mask = mask[:, : logits.shape[1]]
        loss = masked_bce_loss(logits, labels, mask, criterion)

        batch_items = int(mask.sum().item())
        total_loss += float(loss.item()) * batch_items
        total_items += batch_items

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

    rows = rows_from_prob_entries(
        entries,
        threshold=args.threshold,
        fold_seconds=fold_seconds,
        filter_size=args.filter_size,
        max_predictions_per_song=args.max_predictions_per_song,
        min_predictions_per_song=args.min_predictions_per_song,
    )
    metrics = evaluate_boundary_predictions(rows)
    return total_loss / max(total_items, 1), metrics, rows, entries


def parse_threshold_grid(value: str) -> list[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if part:
            thresholds.append(float(part))
    return thresholds


def select_best_threshold(
    entries: list[dict],
    args: argparse.Namespace,
    fold_seconds: float,
) -> tuple[float, dict]:
    candidates = parse_threshold_grid(args.threshold_grid)
    if args.threshold not in candidates:
        candidates.append(args.threshold)
    best_threshold = args.threshold
    best_metrics = None
    for threshold in sorted(set(candidates)):
        rows = rows_from_prob_entries(
            entries,
            threshold=threshold,
            fold_seconds=fold_seconds,
            filter_size=args.filter_size,
            max_predictions_per_song=args.max_predictions_per_song,
            min_predictions_per_song=args.min_predictions_per_song,
        )
        metrics = evaluate_boundary_predictions(rows)
        if best_metrics is None or metrics[args.selection_metric]["f1"] > best_metrics[args.selection_metric]["f1"]:
            best_threshold = threshold
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data_root = args.data_root.resolve()
    metadata_path = args.metadata or data_root / "songs_dataset.json"
    if not args.splits.exists():
        create_and_save_splits(metadata_path, args.splits, seed=args.seed)

    run_name = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_dir or Path(__file__).resolve().parent / "artifacts" / "runs" / run_name
    ckpt_dir = args.checkpoint_dir
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    metadata = load_metadata(metadata_path)
    splits = load_splits(args.splits)
    datasets = {
        split: VariationBoundaryDataset(
            metadata=metadata,
            stems=stems,
            mel_dir=args.mel_dir,
            sr=args.sr,
            hop_length=args.hop_length,
            fold_seconds=args.fold_seconds,
            label_tolerance=args.label_tolerance,
            normalize=args.normalize_mel,
        )
        for split, stems in splits["splits"].items()
    }

    missing = datasets["train"].missing_features
    missing += datasets["val"].missing_features
    missing += datasets["test"].missing_features
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} Mel files, e.g. {preview}. "
            "Run: python3 bishe/variation_boundary_repro/extract_mels.py"
        )

    loaders = {
        "train": make_loader(datasets["train"], args, shuffle=True),
        "val": make_loader(datasets["val"], args, shuffle=False),
        "test": make_loader(datasets["test"], args, shuffle=False),
    }

    device = resolve_device(args.device)
    model = SACNFolk(
        n_mels=args.n_mels,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        sr=args.sr,
        hop_length=args.hop_length,
        fold_seconds=args.fold_seconds,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )

    history = []
    best_f1 = -1.0
    bad_epochs = 0
    best_path = ckpt_dir / "best.pt"

    print(f"Device: {device}")
    print(f"Train/val/test: {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            accum_steps=args.accum_steps,
        )
        val_loss, val_metrics, _, val_entries = evaluate(model, loaders["val"], criterion, device, args)
        selected_threshold = args.threshold
        if args.tune_threshold:
            selected_threshold, val_metrics = select_best_threshold(
                val_entries,
                args,
                fold_seconds=model.fold_size * args.hop_length / args.sr,
            )
        val_f1 = val_metrics[args.selection_metric]["f1"]
        scheduler.step(val_f1)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "selected_threshold": selected_threshold,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"{args.selection_metric}F={val_f1:.4f} HR3F={val_metrics['HR3']['f1']:.4f} "
            f"HR3P={val_metrics['HR3']['precision']:.4f} "
            f"HR3R={val_metrics['HR3']['recall']:.4f} "
            f"SEG3F={val_metrics['SEG3']['f1']:.4f} threshold={selected_threshold:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": serialize_args(args),
                    "splits": splits,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selected_threshold": selected_threshold,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        save_json({"history": history}, run_dir / "metrics.json")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    args.threshold = checkpoint.get("selected_threshold", args.threshold)
    test_loss, test_metrics, test_rows, _ = evaluate(model, loaders["test"], criterion, device, args)
    save_json(
        {
            "best_epoch": checkpoint["epoch"],
            "best_val_metrics": checkpoint["val_metrics"],
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "history": history,
            "args": serialize_args(args),
        },
        run_dir / "metrics.json",
    )
    save_json(test_rows, run_dir / "test_predictions.json")
    print(json.dumps({"test_loss": test_loss, "test_metrics": test_metrics}, indent=2))
    print(f"Best checkpoint: {best_path}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
