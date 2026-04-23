#!/usr/bin/env python3
"""Extract frame-level 24-dim features (12 chroma + 12 delta-chroma) from wav files.

Output for each song: <feature_dir>/<filename>.npy with shape (24, T), dtype=float32.
This format is compatible with model_st.py + ChromaDataset.
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
    parser = argparse.ArgumentParser(description="Extract 24-dim frame features from wav files.")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--dataset-json", default="songs_dataset.json", help="Path to songs_dataset.json")
    parser.add_argument("--feature-dir", default="features_chroma24", help="Output feature directory")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length")
    parser.add_argument("--n-chroma", type=int, default=12, help="Number of chroma bins")
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

    for record in tqdm(records, desc="extract", unit="song"):
        file_id = str(record["filename"]).strip()
        audio_path = (root / record["audio_path"]).resolve()
        out_path = out_dir / f"{file_id}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            y, sr = librosa.load(audio_path.as_posix(), sr=args.sr, mono=True)
            chroma = librosa.feature.chroma_stft(
                y=y,
                sr=sr,
                hop_length=args.hop_length,
                n_chroma=args.n_chroma,
            )
            delta = librosa.feature.delta(chroma, order=1)
            feat = np.concatenate([chroma, delta], axis=0).astype(np.float32)

            if feat.ndim != 2 or feat.shape[0] != args.n_chroma * 2:
                raise ValueError(f"unexpected feature shape {feat.shape} for {file_id}")

            np.save(out_path, feat)
            ok += 1
        except Exception as err:  # pragma: no cover
            failed += 1
            print(f"[ERROR] {file_id}: {err}")

    print("=== Feature Extraction Summary ===")
    print(f"records: {len(records)}")
    print(f"saved: {ok}")
    print(f"skipped_existing: {skipped}")
    print(f"failed: {failed}")
    print(f"feature_dir: {out_dir}")


if __name__ == "__main__":
    main()
