#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.dataset import normalize_mel
from src.model import SACNFolk
from src.postprocess import process_prob_sections
from src.utils import default_data_root, load_metadata, save_json


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Predict variation boundaries.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--mel-dir", type=Path, default=root / "artifacts" / "mels")
    parser.add_argument("--checkpoint", type=Path, default=root / "artifacts" / "checkpoints" / "best.pt")
    parser.add_argument("--filename", type=str, default=None, help="Stem such as 99hh1. Predict all if omitted.")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "predictions.json")
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument("--filter-size", type=int, default=9)
    parser.add_argument("--max-predictions-per-song", type=int, default=0)
    parser.add_argument("--min-predictions-per-song", type=int, default=0)
    parser.add_argument("--no-normalize-mel", dest="normalize_mel", action="store_false")
    parser.add_argument("--normalize-mel", dest="normalize_mel", action="store_true")
    parser.set_defaults(normalize_mel=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


@torch.no_grad()
def predict_one(model: SACNFolk, mel_path: Path, args: argparse.Namespace, normalize: bool) -> list[float]:
    mel = np.load(mel_path).astype(np.float32).T
    if normalize:
        mel = normalize_mel(mel)
    x = torch.from_numpy(mel).unsqueeze(0).to(args.device)
    probs = torch.sigmoid(model(x, frame_lengths=[mel.shape[0]]))[0].cpu()
    return process_prob_sections(
        probs,
        fold_seconds=model.fold_size * model.hop_length / model.sr,
        filter_size=args.filter_size,
        threshold=args.threshold,
        max_predictions=args.max_predictions_per_song if args.max_predictions_per_song > 0 else None,
        min_predictions=args.min_predictions_per_song,
    )


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata or args.data_root.resolve() / "songs_dataset.json"
    metadata = load_metadata(metadata_path)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = checkpoint["args"]
    normalize = ckpt_args.get("normalize_mel", True) if args.normalize_mel is None else args.normalize_mel

    model = SACNFolk(
        n_mels=ckpt_args["n_mels"],
        embed_dim=ckpt_args["embed_dim"],
        hidden_size=ckpt_args["hidden_size"],
        lstm_layers=ckpt_args["lstm_layers"],
        dropout=ckpt_args["dropout"],
        sr=ckpt_args["sr"],
        hop_length=ckpt_args["hop_length"],
        fold_seconds=ckpt_args["fold_seconds"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    for item in metadata:
        stem = item["filename"]
        if args.filename and stem != args.filename:
            continue
        pred_times = predict_one(model, args.mel_dir / f"{stem}.npy", args, normalize=normalize)
        rows.append(
            {
                "filename": stem,
                "pred_times": pred_times,
                "true_times": item.get("boundary_times", []),
            }
        )
    save_json(rows, args.output)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
