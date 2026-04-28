#!/usr/bin/env bash

# 一键对比脚本（当前版本）：
# 固定 w_seconds=1.0、smooth-kernel=11，扫描 peak-threshold，
# 并支持多个随机种子重复实验。
#
# 默认扫描：
#   threshold: 0.003 0.004 0.005 0.006 0.008
#   seed:      42 43 44
#
# 用法：
#   bash scripts/run_compare_w_th.sh
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
  --w-seconds 1.0
  --sigma 24
  --hidden-size 128
  --num-layers 2
  --tolerance 3
  --smooth-kernel 11
  --early-stop-patience 20
  --epochs 60
  --export-boundary-diff
)

# 扫描组合
THRESHOLD_LIST=("0.003" "0.004" "0.005" "0.006" "0.008")
SEED_LIST=("42" "43" "44")

TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="runs/compare_th_k11_${TS}"
mkdir -p "$EXP_ROOT"

echo "=== 参数对比开始 ==="
echo "project_root: $(pwd)"
echo "exp_root:     $EXP_ROOT"
echo "fixed:        w_seconds=1.0, smooth_kernel=11"
echo "thresholds:   ${THRESHOLD_LIST[*]}"
echo "seeds:        ${SEED_LIST[*]}"
echo

run_idx=0
for th in "${THRESHOLD_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    run_idx=$((run_idx + 1))
    th_tag="${th/./p}"
    run_dir="${EXP_ROOT}/run_${run_idx}_th${th_tag}_seed${seed}"

    echo "---- [${run_idx}] threshold=${th}, seed=${seed} ----"
    echo "run_dir: ${run_dir}"

    if python "$TRAIN_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --peak-threshold "$th" \
      --seed "$seed" \
      --run-dir "$run_dir"; then
      echo "[OK] run_${run_idx}"
    else
      echo "[FAIL] run_${run_idx}" | tee -a "${EXP_ROOT}/failed_runs.txt"
    fi
    echo
  done
done

python3 - "$EXP_ROOT" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
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
                "seed": "",
                "w_seconds": "",
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

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rep = json.loads(final_path.read_text(encoding="utf-8"))
    args = cfg.get("args", {})
    test = rep.get("test_metrics", {})
    rows.append(
        {
            "run_dir": run_dir.name,
            "status": "ok",
            "seed": args.get("seed", ""),
            "w_seconds": args.get("w_seconds", ""),
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

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None

# 1) 逐run明细（按 test_hr3f 排序）
rows_sorted = sorted(rows, key=lambda r: _to_float(r.get("test_hr3f")) or -1.0, reverse=True)
summary_csv = exp_root / "summary.csv"
fieldnames = [
    "run_dir",
    "status",
    "seed",
    "w_seconds",
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
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_sorted)

# 2) 按 threshold 聚合（跨seed均值/标准差）
groups = defaultdict(list)
for r in rows:
    if r.get("status") != "ok":
        continue
    th = str(r.get("peak_threshold"))
    groups[th].append(r)

agg_rows = []
for th, rs in groups.items():
    def collect(key):
        vals = [_to_float(x.get(key)) for x in rs]
        vals = [v for v in vals if v is not None]
        if not vals:
            return "", ""
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    hr3f_mean, hr3f_std = collect("test_hr3f")
    hr3p_mean, hr3p_std = collect("test_hr3p")
    hr3r_mean, hr3r_std = collect("test_hr3r")
    hr05f_mean, hr05f_std = collect("test_hr05f")
    loss_mean, loss_std = collect("test_loss")
    peaks_mean, peaks_std = collect("avg_pred_peaks")

    agg_rows.append(
        {
            "peak_threshold": th,
            "num_runs": len(rs),
            "test_hr3f_mean": hr3f_mean,
            "test_hr3f_std": hr3f_std,
            "test_hr3p_mean": hr3p_mean,
            "test_hr3p_std": hr3p_std,
            "test_hr3r_mean": hr3r_mean,
            "test_hr3r_std": hr3r_std,
            "test_hr05f_mean": hr05f_mean,
            "test_hr05f_std": hr05f_std,
            "test_loss_mean": loss_mean,
            "test_loss_std": loss_std,
            "avg_pred_peaks_mean": peaks_mean,
            "avg_pred_peaks_std": peaks_std,
        }
    )

agg_rows.sort(key=lambda r: _to_float(r.get("test_hr3f_mean")) or -1.0, reverse=True)
agg_csv = exp_root / "summary_by_threshold.csv"
agg_fields = [
    "peak_threshold",
    "num_runs",
    "test_hr3f_mean",
    "test_hr3f_std",
    "test_hr3p_mean",
    "test_hr3p_std",
    "test_hr3r_mean",
    "test_hr3r_std",
    "test_hr05f_mean",
    "test_hr05f_std",
    "test_loss_mean",
    "test_loss_std",
    "avg_pred_peaks_mean",
    "avg_pred_peaks_std",
]
with agg_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=agg_fields)
    writer.writeheader()
    writer.writerows(agg_rows)

print(f"summary_csv: {summary_csv}")
print(f"summary_by_threshold_csv: {agg_csv}")
print("top thresholds by mean test_hr3f:")
for r in agg_rows[:5]:
    print(
        f"- th={r['peak_threshold']}: "
        f"HR3F={r['test_hr3f_mean']:.6f}±{r['test_hr3f_std']:.6f}, "
        f"HR3P={r['test_hr3p_mean']:.6f}, HR3R={r['test_hr3r_mean']:.6f}, "
        f"HR.5F={r['test_hr05f_mean']:.6f}, loss={r['test_loss_mean']:.6f}"
    )
PY

echo
echo "=== 参数对比完成 ==="
echo "明细汇总: ${EXP_ROOT}/summary.csv"
echo "阈值聚合: ${EXP_ROOT}/summary_by_threshold.csv"
