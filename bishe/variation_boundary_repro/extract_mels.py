#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import librosa
import numpy as np
from tqdm import tqdm

from src.utils import ensure_dir, load_metadata, default_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract log Mel spectrograms.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_items(metadata_path: Path) -> Iterable[dict]:
    yield from load_metadata(metadata_path)


def extract_one(item: dict, data_root: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    wav_path = data_root / item["audio_path"]
    out_path = output_dir / f"{item['filename']}.npy"
    if out_path.exists() and not args.overwrite:
        return out_path

    y, sr = librosa.load(wav_path, sr=args.sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmax=args.fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    np.save(out_path, mel_db)
    return out_path


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    metadata = args.metadata or data_root / "songs_dataset.json"
    output_dir = args.output_dir or Path(__file__).resolve().parent / "artifacts" / "mels"
    ensure_dir(output_dir)

    items = list(iter_items(metadata))
    for item in tqdm(items, desc="extract mels"):
        extract_one(item, data_root, output_dir, args)
    print(f"Saved Mel features to {output_dir}")


if __name__ == "__main__":
    main()
