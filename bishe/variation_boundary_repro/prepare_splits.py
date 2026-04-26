#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.splits import create_and_save_splits
from src.utils import default_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grouped 8:1:1 splits.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    metadata = args.metadata or data_root / "songs_dataset.json"
    output = args.output or Path(__file__).resolve().parent / "artifacts" / "splits.json"
    summary = create_and_save_splits(metadata, output, seed=args.seed)
    print(f"Saved splits to {output}")
    for split, count in summary["counts"].items():
        print(f"{split}: {count} files, {summary['group_counts'][split]} groups")


if __name__ == "__main__":
    main()
