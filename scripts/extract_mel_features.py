#!/usr/bin/env python3
"""Extract log-Mel spectrogram features from wav files.

Output for each song:
  <feature_dir>/<filename>.npy with shape (n_mels, T), dtype=float32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import librosa
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "librosa is required. Please install dependencies first, e.g. `pip install librosa soundfile`"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract log-Mel spectrogram features from wav files.")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--dataset-json", default="songs_dataset.json", help="Path to songs_dataset.json")
    parser.add_argument("--feature-dir", default="features_mel81", help="Output feature directory")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length")
    parser.add_argument("--n-mels", type=int, default=81, help="Number of Mel bins")
    parser.add_argument("--fmin", type=float, default=30.0, help="Minimum frequency for Mel filter")
    parser.add_argument("--fmax", type=float, default=16000.0, help="Maximum frequency for Mel filter")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    dataset_path = (root / args.dataset_json).resolve()
    out_dir = (root / args.feature_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    ok = 0
    skipped = 0
    failed = 0

    for record in tqdm(records, desc="extract-mel", unit="song"):
        file_id = str(record["filename"]).strip()
        audio_path = (root / record["audio_path"]).resolve()
        out_path = out_dir / f"{file_id}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            y, sr = librosa.load(audio_path.as_posix(), sr=args.sr, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax,
                power=2.0,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

            if mel_db.ndim != 2 or mel_db.shape[0] != args.n_mels:
                raise ValueError(f"unexpected mel shape {mel_db.shape} for {file_id}")

            np.save(out_path, mel_db)
            ok += 1
        except Exception as err:  # pragma: no cover
            failed += 1
            print(f"[ERROR] {file_id}: {err}")

    print("=== Mel Feature Extraction Summary ===")
    print(f"records: {len(records)}")
    print(f"saved: {ok}")
    print(f"skipped_existing: {skipped}")
    print(f"failed: {failed}")
    print(f"feature_dir: {out_dir}")


if __name__ == "__main__":
    main()

