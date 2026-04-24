#!/usr/bin/env bash
set -euo pipefail

# Run SA-CNFolk with Table III parameter combinations.
# Note: Table III in the paper includes one duplicated row (1.0, 24, 128, 2).

ROOT_DIR="${1:-.}"

COMMON_ARGS=(
  --root "${ROOT_DIR}"
  --dataset-json songs_dataset.json
  --feature-dir features_mel81
  --auto-extract-mel
  --n-mels 81
  --batch-size 1
  --tolerance 3
  --early-stop-patience 5
  --epochs 80
)

python scripts/train_sa_cnfolk.py "${COMMON_ARGS[@]}" --w-seconds 1.0  --sigma 24 --hidden-size 128 --num-layers 2
python scripts/train_sa_cnfolk.py "${COMMON_ARGS[@]}" --w-seconds 0.25 --sigma 24 --hidden-size 64  --num-layers 2
python scripts/train_sa_cnfolk.py "${COMMON_ARGS[@]}" --w-seconds 1.0  --sigma 24 --hidden-size 128 --num-layers 2
python scripts/train_sa_cnfolk.py "${COMMON_ARGS[@]}" --w-seconds 0.5  --sigma 12 --hidden-size 64  --num-layers 3
python scripts/train_sa_cnfolk.py "${COMMON_ARGS[@]}" --w-seconds 1.0  --sigma 24 --hidden-size 256 --num-layers 3

