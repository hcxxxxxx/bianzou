#!/usr/bin/env bash

# 基于 log_author.txt 的高分组合做复现实验（当前数据与代码管线）
# 说明：
# 1) 组合来自作者日志中 F1 较高的参数（去重后手工筛选）。
# 2) 后处理参数默认固定（可按需改）：smooth-kernel=11, peak-threshold=0.005。
# 3) 每组写入独立 run_dir，并自动汇总 summary.csv。
#
# 用法：
#   bash scripts/run_author_top_combos.sh
#   bash scripts/run_author_top_combos.sh /path/to/project_root
#
# 可选环境变量（不设置则使用默认值）：
#   PEAK_THRESHOLD=0.005
#   SMOOTH_KERNEL=11
#   SEED=42

set -euo pipefail

PROJECT_ROOT="${1:-.}"
cd "$PROJECT_ROOT"

TRAIN_SCRIPT="scripts/train_sa_cnfolk.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[ERROR] 找不到训练脚本: $TRAIN_SCRIPT"
  exit 1
fi

PEAK_THRESHOLD="${PEAK_THRESHOLD:-0.005}"
SMOOTH_KERNEL="${SMOOTH_KERNEL:-11}"
SEED="${SEED:-42}"

# 你当前常用的基础参数
COMMON_ARGS=(
  --root .
  --dataset-json songs_dataset.json
  --feature-dir features_mel128_f8k
  --auto-extract-mel
  --n-mels 128
  --fmax 8000
  --batch-size 1
  --tolerance 3
  --early-stop-patience 20
  --epochs 60
  --export-boundary-diff
  --seed "$SEED"
  --smooth-kernel "$SMOOTH_KERNEL"
  --peak-threshold "$PEAK_THRESHOLD"
)

# 组合格式：w_seconds sigma hidden_size num_layers
# 包含作者日志高分组合 + 当前主配置(1.0,24,128,2)作为对照
COMBOS=(
  "0.25 24 64 2"
  "0.5 12 64 3"
  "0.5 12 128 2"
  "0.25 12 256 3"
  "0.5 6 64 2"
  "0.5 6 128 2"
  "0.5 12 64 2"
  "1.0 12 128 2"
  "1.0 24 128 2"
)

TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="runs/author_top_combos_${TS}"
mkdir -p "$EXP_ROOT"

echo "=== 作者组合复跑开始 ==="
echo "project_root:   $(pwd)"
echo "exp_root:       $EXP_ROOT"
echo "fixed seed:     $SEED"
echo "fixed postproc: smooth_kernel=$SMOOTH_KERNEL, peak_threshold=$PEAK_THRESHOLD"
echo

run_idx=0
for cfg in "${COMBOS[@]}"; do
  read -r w sigma hidden layers <<<"$cfg"
  run_idx=$((run_idx + 1))
  w_tag="${w/./p}"
  run_dir="${EXP_ROOT}/run_${run_idx}_w${w_tag}_s${sigma}_h${hidden}_l${layers}"

  echo "---- [${run_idx}] w=${w}, sigma=${sigma}, hidden=${hidden}, layers=${layers} ----"
  echo "run_dir: ${run_dir}"

  if python "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --w-seconds "$w" \
    --sigma "$sigma" \
    --hidden-size "$hidden" \
    --num-layers "$layers" \
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
    cfg_path = run_dir / "config.json"
    rep_path = run_dir / "final_report.json"
    if not cfg_path.exists() or not rep_path.exists():
        rows.append(
            {
                "run_dir": run_dir.name,
                "status": "missing_report",
                "w_seconds": "",
                "sigma": "",
                "hidden_size": "",
                "num_layers": "",
                "smooth_kernel": "",
                "peak_threshold": "",
                "best_epoch": "",
                "best_val_hr3f": "",
                "test_hr3f": "",
                "test_hr3p": "",
                "test_hr3r": "",
                "test_hr05f": "",
                "test_loss": "",
                "avg_pred_peaks": "",
            }
        )
        continue

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    args = cfg.get("args", {})
    test = rep.get("test_metrics", {})

    rows.append(
        {
            "run_dir": run_dir.name,
            "status": "ok",
            "w_seconds": args.get("w_seconds", ""),
            "sigma": args.get("sigma", ""),
            "hidden_size": args.get("hidden_size", ""),
            "num_layers": args.get("num_layers", ""),
            "smooth_kernel": args.get("smooth_kernel", ""),
            "peak_threshold": args.get("peak_threshold", ""),
            "best_epoch": rep.get("best_epoch", ""),
            "best_val_hr3f": rep.get("best_hr3f", ""),
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
summary = exp_root / "summary.csv"
fieldnames = [
    "run_dir",
    "status",
    "w_seconds",
    "sigma",
    "hidden_size",
    "num_layers",
    "smooth_kernel",
    "peak_threshold",
    "best_epoch",
    "best_val_hr3f",
    "test_hr3f",
    "test_hr3p",
    "test_hr3r",
    "test_hr05f",
    "test_loss",
    "avg_pred_peaks",
]
with summary.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"summary_csv: {summary}")
print("top runs by test_hr3f:")
for r in rows[:8]:
    print(
        f"- {r['run_dir']}: "
        f"w={r['w_seconds']}, sigma={r['sigma']}, h={r['hidden_size']}, l={r['num_layers']}, "
        f"HR3F={r['test_hr3f']}, HR3P={r['test_hr3p']}, HR3R={r['test_hr3r']}, HR.5F={r['test_hr05f']}"
    )
PY

echo
echo "=== 作者组合复跑完成 ==="
echo "汇总文件: ${EXP_ROOT}/summary.csv"
