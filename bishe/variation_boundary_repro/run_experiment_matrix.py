#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of reproduction experiments.")
    parser.add_argument("--preset", choices=["focused", "full"], default="focused")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "experiment_matrix")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N experiments; 0 means all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-train-arg", action="append", default=[], help="Append one raw argument to train.py; repeatable.")
    return parser.parse_args()


def focused_experiments() -> list[dict]:
    return [
        {
            "name": "author_baseline",
            "threshold": 0.0001,
            "label_tolerance": 3.0,
            "normalize_mel": True,
            "model_variant": "cnn_lstm",
            "peak_mode": "author",
            "time_position": "center",
        },
        {
            "name": "author_threshold_001",
            "threshold": 0.001,
            "label_tolerance": 3.0,
            "normalize_mel": True,
            "model_variant": "cnn_lstm",
            "peak_mode": "author",
            "time_position": "center",
        },
        {
            "name": "time_start",
            "threshold": 0.0001,
            "label_tolerance": 3.0,
            "normalize_mel": True,
            "model_variant": "cnn_lstm",
            "peak_mode": "author",
            "time_position": "start",
        },
        {
            "name": "strict_peak",
            "threshold": 0.0001,
            "label_tolerance": 3.0,
            "normalize_mel": True,
            "model_variant": "cnn_lstm",
            "peak_mode": "strict",
            "time_position": "center",
        },
        {
            "name": "no_normalize",
            "threshold": 0.0001,
            "label_tolerance": 3.0,
            "normalize_mel": False,
            "model_variant": "cnn_lstm",
            "peak_mode": "author",
            "time_position": "center",
        },
        {
            "name": "label_tolerance_2",
            "threshold": 0.0001,
            "label_tolerance": 2.0,
            "normalize_mel": True,
            "model_variant": "cnn_lstm",
            "peak_mode": "author",
            "time_position": "center",
        },
        {
            "name": "mel_lstm",
            "threshold": 0.0001,
            "label_tolerance": 3.0,
            "normalize_mel": True,
            "model_variant": "mel_lstm",
            "peak_mode": "author",
            "time_position": "center",
        },
        {
            "name": "mel_lstm_threshold_001",
            "threshold": 0.001,
            "label_tolerance": 3.0,
            "normalize_mel": True,
            "model_variant": "mel_lstm",
            "peak_mode": "author",
            "time_position": "center",
        },
    ]


def full_experiments() -> list[dict]:
    experiments = []
    grid = {
        "threshold": [0.0001, 0.001],
        "label_tolerance": [3.0, 2.0],
        "normalize_mel": [True, False],
        "model_variant": ["cnn_lstm", "mel_lstm"],
        "peak_mode": ["author", "strict"],
        "time_position": ["center", "start"],
    }
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        exp = dict(zip(keys, values))
        norm = "norm" if exp["normalize_mel"] else "rawmel"
        exp["name"] = (
            f"{exp['model_variant']}_{norm}_tol{exp['label_tolerance']:g}_"
            f"thr{exp['threshold']:g}_{exp['peak_mode']}_{exp['time_position']}"
        ).replace(".", "p")
        experiments.append(exp)
    return experiments


def train_command(exp: dict, args: argparse.Namespace, output_root: Path) -> list[str]:
    run_dir = output_root / "runs" / exp["name"]
    checkpoint_dir = output_root / "checkpoints" / exp["name"]
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--batch-size",
        "1",
        "--filter-size",
        "9",
        "--threshold",
        str(exp["threshold"]),
        "--label-tolerance",
        str(exp["label_tolerance"]),
        "--model-variant",
        exp["model_variant"],
        "--peak-mode",
        exp["peak_mode"],
        "--time-position",
        exp["time_position"],
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--run-dir",
        str(run_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if not exp["normalize_mel"]:
        cmd.append("--no-normalize-mel")
    cmd.extend(args.extra_train_arg)
    return cmd


def read_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_result(exp: dict, status: str, elapsed: float, metrics: dict) -> dict:
    row = {**exp, "status": status, "elapsed_sec": round(elapsed, 3)}
    row["best_epoch"] = metrics.get("best_epoch")
    for prefix, source in [
        ("val", metrics.get("best_val_metrics", {})),
        ("test", metrics.get("test_metrics", {})),
    ]:
        for metric_name in ["HR.5", "HR3", "SEG3"]:
            values = source.get(metric_name, {})
            safe_name = metric_name.replace(".", "")
            for key in ["precision", "recall", "f1", "predicted", "truth"]:
                row[f"{prefix}_{safe_name}_{key}"] = values.get(key)
    return row


def write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    preferred = [
        "name",
        "status",
        "elapsed_sec",
        "best_epoch",
        "threshold",
        "label_tolerance",
        "normalize_mel",
        "model_variant",
        "peak_mode",
        "time_position",
        "val_HR3_f1",
        "val_HR3_precision",
        "val_HR3_recall",
        "test_HR3_f1",
        "test_HR3_precision",
        "test_HR3_recall",
        "test_HR5_f1",
        "test_SEG3_f1",
    ]
    ordered = preferred + [key for key in keys if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_root = args.output_root / timestamp
    experiments = focused_experiments() if args.preset == "focused" else full_experiments()
    if args.limit > 0:
        experiments = experiments[: args.limit]

    rows = []
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {output_root}")
    print(f"Experiments: {len(experiments)}")

    for index, exp in enumerate(experiments, start=1):
        print(f"\n[{index}/{len(experiments)}] {exp['name']}")
        cmd = train_command(exp, args, output_root)
        print(" ".join(cmd))
        if args.dry_run:
            rows.append(flatten_result(exp, "dry_run", 0.0, {}))
            continue

        start = time.time()
        status = "ok"
        try:
            subprocess.run(cmd, cwd=ROOT.parents[1], check=True)
        except subprocess.CalledProcessError as exc:
            status = f"failed:{exc.returncode}"
        elapsed = time.time() - start
        metrics = read_metrics(output_root / "runs" / exp["name"])
        rows.append(flatten_result(exp, status, elapsed, metrics))
        write_summary(rows, output_root / "summary.csv")
        with (output_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    write_summary(rows, output_root / "summary.csv")
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nSummary: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()

