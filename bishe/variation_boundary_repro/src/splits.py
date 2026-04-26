from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

from .utils import ensure_dir, load_json, load_metadata, save_json


TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$")


def song_group_key(filename: str) -> str:
    stem = Path(filename).stem
    match = TRAILING_NUMBER.match(stem)
    return match.group(1) if match else stem


def grouped_items(metadata: list[dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in metadata:
        stem = item["filename"]
        groups[song_group_key(stem)].append(stem)
    return {key: sorted(value) for key, value in groups.items()}


def group_split(
    metadata: list[dict],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict:
    groups = grouped_items(metadata)
    rng = random.Random(seed)
    group_rows = list(groups.items())
    rng.shuffle(group_rows)
    group_rows.sort(key=lambda item: len(item[1]), reverse=True)

    total = sum(len(v) for _, v in group_rows)
    targets = {
        "train": total * ratios[0],
        "val": total * ratios[1],
        "test": total * ratios[2],
    }
    splits = {"train": [], "val": [], "test": []}
    split_groups = {"train": [], "val": [], "test": []}

    for group, stems in group_rows:
        deficits = {
            split: targets[split] - len(splits[split])
            for split in splits
        }
        candidates = sorted(deficits, key=lambda split: deficits[split], reverse=True)
        chosen = candidates[0]
        splits[chosen].extend(stems)
        split_groups[chosen].append(group)

    for split in splits:
        splits[split] = sorted(splits[split])
        split_groups[split] = sorted(split_groups[split])

    return {
        "seed": seed,
        "ratios": ratios,
        "splits": splits,
        "groups": split_groups,
        "counts": {split: len(stems) for split, stems in splits.items()},
        "group_counts": {split: len(groups_) for split, groups_ in split_groups.items()},
    }


def assert_no_group_leakage(split_obj: dict) -> None:
    owner = {}
    for split, groups in split_obj["groups"].items():
        for group in groups:
            if group in owner:
                raise ValueError(f"group leakage: {group} in {owner[group]} and {split}")
            owner[group] = split


def create_and_save_splits(metadata_path: Path, output_path: Path, seed: int = 42) -> dict:
    metadata = load_metadata(metadata_path)
    split_obj = group_split(metadata, seed=seed)
    assert_no_group_leakage(split_obj)
    ensure_dir(output_path.parent)
    save_json(split_obj, output_path)
    return split_obj


def load_splits(path: Path) -> dict:
    split_obj = load_json(path)
    assert_no_group_leakage(split_obj)
    return split_obj

