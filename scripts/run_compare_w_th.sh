#!/usr/bin/env bash

# 一键对比脚本：
# 固定 w_seconds=1.0、peak-threshold=0.005，批量扫描 smooth-kernel 组合，
# 每组实验写入独立 run_dir，并在结束后汇总结果到 summary.csv。
#
# 用法：
#   bash scripts/run_compare_w_th.sh
# 可选：
#   bash scripts/run_compare_w_th.sh /path/to/project_root

set -euo pipefail

PROJECT_ROOT="${1:-.}"
cd "$PROJECT_ROOT"

TRAIN_SCRIPT="scripts/train_sa_cnfolk.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[ERROR] 找不到训练脚本: $TRAIN_SCRIPT"
  exit 1
fi

# 你当前常用的基础参数
COMMON_ARGS=(
  --root .
  --dataset-json songs_dataset.json
  --feature-dir features_mel128_f8k
  --auto-extract-mel
  --n-mels 128
  --fmax 8000
  --batch-size 1
  --sigma 24
  --hidden-size 128
  --num-layers 2
  --tolerance 3
  --smooth-kernel 9
  --early-stop-patience 20
  --epochs 60
  --export-boundary-diff
)

# 固定参数
FIXED_W_SECONDS="1.0"
FIXED_PEAK_THRESHOLD="0.005"

# 扫描组合
KERNEL_LIST=("7" "9" "11" "13")

TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="runs/compare_w_th_${TS}"
mkdir -p "$EXP_ROOT"

echo "=== 参数对比开始 ==="
echo "project_root: $(pwd)"
echo "exp_root:     $EXP_ROOT"
echo

run_idx=0
for k in "${KERNEL_LIST[@]}"; do
  run_idx=$((run_idx + 1))
  w_tag="${FIXED_W_SECONDS/./p}"
  th_tag="${FIXED_PEAK_THRESHOLD/./p}"
  run_dir="${EXP_ROOT}/run_${run_idx}_w${w_tag}_th${th_tag}_k${k}"

  echo "---- [${run_idx}] w_seconds=${FIXED_W_SECONDS}, peak_threshold=${FIXED_PEAK_THRESHOLD}, smooth_kernel=${k} ----"
  echo "run_dir: ${run_dir}"

  if python "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --w-seconds "$FIXED_W_SECONDS" \
    --peak-threshold "$FIXED_PEAK_THRESHOLD" \
    --smooth-kernel "$k" \
    --run-dir "$run_dir"; then
    echo "[OK] run_${run_idx}"
  else
    echo "[FAIL] run_${run_idx}" | tee -a "${EXP_ROOT}/failed_runs.txt"
  fi
  echo
done

python3 - "$EXP_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

exp_root = Path(sys.argv[1])
rows = []

for run_dir in sorted(exp_root.glob("run_*")):
    config_path = run_dir / "config.json"
    final_path = run_dir / "final_report.json"
    if not config_path.exists() or not final_path.exists():
        rows.append(
            {
                "run_dir": run_dir.name,
                "status": "missing_report",
                "w_seconds": "",
                "peak_threshold": "",
                "best_epoch": "",
                "best_val_hr3f": "",
                "smooth_kernel": "",
                "test_hr3f": "",
                "test_hr3p": "",
                "test_hr3r": "",
                "test_hr05f": "",
                "test_loss": "",
                "avg_pred_peaks": "",
            }
        )
        continue

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rep = json.loads(final_path.read_text(encoding="utf-8"))
    args = cfg.get("args", {})
    test = rep.get("test_metrics", {})
    rows.append(
        {
            "run_dir": run_dir.name,
            "status": "ok",
            "w_seconds": args.get("w_seconds", ""),
            "peak_threshold": args.get("peak_threshold", ""),
            "best_epoch": rep.get("best_epoch", ""),
            "best_val_hr3f": rep.get("best_hr3f", ""),
            "smooth_kernel": args.get("smooth_kernel", ""),
            "test_hr3f": test.get("hr3f", ""),
            "test_hr3p": test.get("hr3p", ""),
            "test_hr3r": test.get("hr3r", ""),
            "test_hr05f": test.get("hr05f", ""),
            "test_loss": test.get("loss", ""),
            "avg_pred_peaks": test.get("avg_pred_peaks", ""),
        }
    )

def key_fn(r):
    try:
        return float(r["test_hr3f"])
    except Exception:
        return -1.0

rows.sort(key=key_fn, reverse=True)

summary_csv = exp_root / "summary.csv"
fieldnames = [
    "run_dir",
    "status",
    "w_seconds",
    "peak_threshold",
    "best_epoch",
    "best_val_hr3f",
    "smooth_kernel",
    "test_hr3f",
    "test_hr3p",
    "test_hr3r",
    "test_hr05f",
    "test_loss",
    "avg_pred_peaks",
]

with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"summary_csv: {summary_csv}")
print("top runs by test_hr3f:")
for r in rows[:5]:
    print(
        f"- {r['run_dir']}: "
        f"w={r['w_seconds']}, th={r['peak_threshold']}, k={r['smooth_kernel']}, "
        f"HR3F={r['test_hr3f']}, HR3P={r['test_hr3p']}, HR3R={r['test_hr3r']}, HR.5F={r['test_hr05f']}"
    )
PY

echo
echo "=== 参数对比完成 ==="
echo "汇总文件: ${EXP_ROOT}/summary.csv"
