#!/usr/bin/env python3
"""基于论文描述实现的 SA-CNFolk 训练流程。

流程：
1) wav -> log-Mel 特征（可选自动提取）
2) 特征嵌入（CNN）
3) 特征聚合（非重叠窗口 w）
4) Bi-LSTM + 线性分类器
5) BCEWithLogitsLoss 训练
6) 后处理（平滑 + 局部极大值）
7) HR3/HR.5 评估

已实现约束：
- 数据划分比例 train:val:test = 8:1:1
- 同一首歌的不同版本只会出现在同一个划分中
- batch_size 必须为 1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SA-CNFolk from paper-style pipeline.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--dataset-json", default="songs_dataset.json", help="Dataset JSON path")
    parser.add_argument("--feature-dir", default="features_mel81", help="Precomputed Mel directory")

    parser.add_argument("--batch-size", type=int, default=1, help="Must be 1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr-plateau-patience", type=int, default=10)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    # 与论文相关的参数
    parser.add_argument("--w-seconds", type=float, default=1.0, help="Feature aggregation window size w (seconds)")
    parser.add_argument("--sigma", type=int, default=24, help="Embedding output channels sigma")
    parser.add_argument("--hidden-size", type=int, default=128, help="Bi-LSTM hidden size h")
    parser.add_argument("--num-layers", type=int, default=2, help="Bi-LSTM layers l")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Frame-label tolerance in seconds")

    # 音频 / Mel 提取参数
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--n-mels", type=int, default=81)
    parser.add_argument("--fmin", type=float, default=30.0)
    parser.add_argument("--fmax", type=float, default=16000.0)
    parser.add_argument("--auto-extract-mel", action="store_true", help="Extract missing Mel files from wav")
    parser.add_argument("--overwrite-mel", action="store_true", help="Re-extract Mel even if files exist")

    # 后处理参数
    parser.add_argument("--smooth-kernel", type=int, default=9, help="Local-max filter size")
    parser.add_argument("--peak-threshold", type=float, default=0.05, help="Local-max threshold")
    parser.add_argument(
        "--export-boundary-diff",
        action="store_true",
        help="导出测试集逐首歌的边界预测误差明细（JSON + CSV）",
    )

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--disable-cudnn", action="store_true", help="Disable cuDNN for debugging/stability")
    parser.add_argument(
        "--debug-batch-crash",
        action="store_true",
        help="On exception, dump current batch/tensor diagnostics to run_dir/crash_debug/*.json",
    )
    parser.add_argument("--run-dir", default="", help="Optional output run directory")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def tensor_stats(t: torch.Tensor | None) -> dict[str, Any] | None:
    if t is None:
        return None
    stats: dict[str, Any] = {}
    try:
        td = t.detach()
        stats.update(
            {
                "shape": list(td.shape),
                "dtype": str(td.dtype),
                "device": str(td.device),
                "numel": int(td.numel()),
            }
        )
    except Exception as exc:
        return {"stats_error": f"detach_or_meta_failed: {type(exc).__name__}: {exc}"}

    if int(stats.get("numel", 0)) == 0:
        stats.update({"min": None, "max": None, "mean": None, "nan_count": 0, "inf_count": 0})
        return stats

    try:
        # 仅在 CPU 上计算统计值；若 CUDA 上下文已损坏，
        # 则返回元数据与错误信息，而不是再次抛异常。
        tf = td.float().to("cpu", copy=True)
        stats.update(
            {
                "min": float(tf.min().item()),
                "max": float(tf.max().item()),
                "mean": float(tf.mean().item()),
                "nan_count": int(torch.isnan(tf).sum().item()),
                "inf_count": int(torch.isinf(tf).sum().item()),
            }
        )
    except Exception as exc:
        stats.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "nan_count": None,
                "inf_count": None,
                "value_stats_error": f"{type(exc).__name__}: {exc}",
            }
        )
    return stats


def load_records(dataset_json: Path) -> list[dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError("dataset json is empty or invalid")
    return records


def extract_music_id(filename: str) -> str | None:
    """从文件名提取 music_id：优先字母前缀；否则去掉末尾数字。"""
    stem = Path(str(filename)).stem
    m = re.match(r"([a-zA-Z]+)", stem)
    if m:
        return m.group(1)
    # 兼容如 99hh1 这种“数字+字母+数字”命名。
    m = re.match(r"^(.*?)(\d+)$", stem)
    if m and m.group(1):
        return m.group(1)
    return stem if stem else None


def split_by_title(records: list[dict[str, Any]], seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """按作者参考逻辑分层划分：region -> music_id -> 文件集合。"""
    rng = random.Random(seed)

    # 构建: region -> music_id -> [records]
    region_music_records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    dropped_missing_region = 0
    dropped_missing_music_id = 0
    dropped_missing_boundary = 0
    for rec in records:
        boundary_times = rec.get("boundary_times")
        if boundary_times is None:
            dropped_missing_boundary += 1
            continue

        region = rec.get("region")
        music_id = extract_music_id(str(rec.get("filename", "")))
        if not region:
            dropped_missing_region += 1
            continue
        if not music_id:
            dropped_missing_music_id += 1
            continue
        region_music_records[str(region)][music_id].append(rec)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    used_regions = 0
    skipped_regions = 0
    dropped_small_region_records = 0

    for _region, music_to_records in region_music_records.items():
        music_ids = list(music_to_records.keys())
        if len(music_ids) < 4:
            # 参考作者逻辑：跳过不满足条件的地域。
            skipped_regions += 1
            dropped_small_region_records += sum(len(v) for v in music_to_records.values())
            continue

        used_regions += 1
        rng.shuffle(music_ids)
        n = len(music_ids)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train_ids = music_ids[:n_train]
        val_ids = music_ids[n_train : n_train + n_val]
        test_ids = music_ids[n_train + n_val :]

        # 确保 test 每个 region 至少有 1 个 music_id。
        if len(test_ids) == 0:
            if len(val_ids) > 0:
                test_ids.append(val_ids.pop())
            elif len(train_ids) > 0:
                test_ids.append(train_ids.pop())

        for music_id in train_ids:
            train_records.extend(music_to_records[music_id])
        for music_id in val_ids:
            val_records.extend(music_to_records[music_id])
        for music_id in test_ids:
            test_records.extend(music_to_records[music_id])

    splits = {"train": train_records, "val": val_records, "test": test_records}

    # 防止 music_id 跨集合泄漏
    split_music_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for split_name, recs in splits.items():
        for rec in recs:
            music_id = extract_music_id(str(rec.get("filename", "")))
            if music_id:
                split_music_ids[split_name].add(music_id)
    if split_music_ids["train"] & split_music_ids["val"]:
        raise RuntimeError("music_id leakage between train/val")
    if split_music_ids["train"] & split_music_ids["test"]:
        raise RuntimeError("music_id leakage between train/test")
    if split_music_ids["val"] & split_music_ids["test"]:
        raise RuntimeError("music_id leakage between val/test")

    split_meta: dict[str, Any] = {
        "strategy": "region_music_id",
        "regions_total": len(region_music_records),
        "regions_used": used_regions,
        "regions_skipped_lt4": skipped_regions,
        "records_dropped_missing_region": dropped_missing_region,
        "records_dropped_missing_music_id": dropped_missing_music_id,
        "records_dropped_missing_boundary_times": dropped_missing_boundary,
        "records_dropped_small_region": dropped_small_region_records,
    }
    return splits, split_meta


def maybe_extract_mel(records: list[dict[str, Any]], root: Path, feature_dir: Path, args: argparse.Namespace) -> None:
    if not args.auto_extract_mel and not args.overwrite_mel:
        return
    try:
        import librosa
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("librosa is required for --auto-extract-mel") from exc

    feature_dir.mkdir(parents=True, exist_ok=True)
    for rec in tqdm(records, desc="extract-mel", unit="song"):
        fid = str(rec["filename"])
        out = feature_dir / f"{fid}.npy"
        if out.exists() and not args.overwrite_mel:
            continue
        wav = (root / rec["audio_path"]).resolve()
        y, sr = librosa.load(wav.as_posix(), sr=args.sr, mono=True)
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
        np.save(out, mel_db)


def validate_features(records: list[dict[str, Any]], feature_dir: Path, n_mels: int) -> None:
    missing = []
    bad = []
    for rec in records:
        p = feature_dir / f"{rec['filename']}.npy"
        if not p.exists():
            missing.append(p.as_posix())
            continue
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] != n_mels:
            bad.append(f"{p.name}: shape={tuple(arr.shape)}")
    if missing:
        raise FileNotFoundError("Missing Mel feature files, sample:\n" + "\n".join(missing[:10]))
    if bad:
        raise ValueError("Bad Mel feature shape, sample:\n" + "\n".join(bad[:10]))


class MelBoundaryDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        feature_dir: Path,
        sr: int,
        hop_length: int,
        tolerance: float,
        w_seconds: float,
    ):
        self.records = records
        self.feature_dir = feature_dir
        self.sr = sr
        self.hop_length = hop_length
        self.tolerance = tolerance
        self.w_seconds = w_seconds
        self.frame_duration = hop_length / sr
        self.fold_size = int(w_seconds / self.frame_duration)

    def __len__(self) -> int:
        return len(self.records)

    def _frame_labels(self, boundary_times: list[float], n_frames: int) -> np.ndarray:
        y = np.zeros(n_frames, dtype=np.float32)
        delta = int(self.tolerance / self.frame_duration)
        for t in boundary_times:
            c = int(t / self.frame_duration)
            s = max(c - delta, 0)
            e = min(c + delta + 1, n_frames)
            y[s:e] = 1.0
        return y

    def _fold_mean(self, seq: np.ndarray) -> np.ndarray:
        n_folds = seq.shape[0] // self.fold_size
        seq = seq[: n_folds * self.fold_size]
        seq = seq.reshape(n_folds, self.fold_size, -1).mean(axis=1)
        return seq

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        fid = str(rec["filename"])
        feat = np.load(self.feature_dir / f"{fid}.npy")  # (n_mels, T)
        feat = feat.T.astype(np.float32)  # (T, n_mels)

        # 默认假设 boundary_times 已在数据集中预处理完成。
        boundary_times = sorted(float(t) for t in rec["boundary_times"])
        y = self._frame_labels(boundary_times, feat.shape[0])  # (T,)
        y_fold = self._fold_mean(y[:, None]).squeeze(-1).astype(np.float32)  # (T/w,)

        return {
            "x": torch.tensor(feat, dtype=torch.float32),
            "y": torch.tensor(y_fold, dtype=torch.float32),
            "boundary_times": boundary_times,
            "song_id": rec["song_id"],
            "title": rec["title"],
            "filename": rec["filename"],
        }


def collate_batch_size_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("batch_size must be 1")
    return batch[0]


class Head(nn.Module):
    """作者补充的分类头实现（按时间维输出边界 logits）。"""

    def __init__(self, dim_embed: int, num_classes: int, init_confidence: float | None = None):
        super().__init__()
        self.classifier = nn.Linear(dim_embed, num_classes)
        if init_confidence is not None:
            self.reset_parameters(init_confidence)

    def reset_parameters(self, confidence: float) -> None:
        if not (0.0 < confidence < 1.0):
            raise ValueError("init_confidence 必须位于 (0, 1) 区间")
        bias = -torch.log(torch.tensor(1.0 / confidence - 1.0, dtype=self.classifier.bias.dtype))
        with torch.no_grad():
            self.classifier.bias.fill_(float(bias.item()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持 (B, T, E) 或 (B, I, T, E) 两种输入形状。
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, T, E)
        if x.dim() != 4:
            raise ValueError(f"Head 输入维度应为 3 或 4，当前为 {x.dim()}")

        batch, inst, frame, embed = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, frame, inst * embed)
        logits = self.classifier(x).permute(0, 2, 1)
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)  # (B, T)
        return logits


class FeatureEmbedding(nn.Module):
    def __init__(self, in_bins: int, sigma: int, dropout: float = 0.2):
        super().__init__()
        del in_bins  # architecture is convolutional; frequency bins handled dynamically.
        c0 = max(1, sigma // 2)
        self.act = nn.ELU()

        self.conv0 = nn.Conv2d(1, c0, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.drop0 = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(c0, sigma, kernel_size=(1, 12), stride=(1, 1), padding=(0, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(sigma, sigma, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))

        self.norm = nn.LayerNorm(sigma)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, 1, T, F)
        x = self.drop0(self.act(self.pool0(self.conv0(x))))
        x = self.drop1(self.act(self.pool1(self.conv1(x))))
        x = self.act(self.pool2(self.conv2(x)))
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, T, sigma)
        x = self.drop2(self.norm(x))
        return x


class SACNFolkModel(nn.Module):
    def __init__(
        self,
        n_mels: int,
        sigma: int,
        hidden_size: int,
        num_layers: int,
        w_seconds: float,
        sr: int,
        hop_length: int,
    ):
        super().__init__()
        self.frame_duration = hop_length / sr
        self.fold_size = int(w_seconds / self.frame_duration)
        self.w_seconds = w_seconds
        self.sigma = sigma

        self.embedding = FeatureEmbedding(n_mels, sigma=sigma, dropout=0.2)
        self.lstm = nn.LSTM(
            input_size=self.fold_size * sigma,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = Head(dim_embed=2 * hidden_size, num_classes=1)

    def _aggregate_windows(self, emb: torch.Tensor) -> torch.Tensor:
        # emb 形状: (B, T, sigma) -> (B, T/w, w*sigma)
        b, t, c = emb.shape
        n = t // self.fold_size
        emb = emb[:, : n * self.fold_size, :]
        emb = emb.reshape(b, n, self.fold_size * c)
        return emb

    def forward(self, x_tf: torch.Tensor) -> torch.Tensor:
        # 当 batch_size=1 时，x_tf 形状为 (T, F)
        x = x_tf.unsqueeze(0).unsqueeze(0)  # (1, 1, T, F)
        emb = self.embedding(x)  # (1, T, sigma)
        agg = self._aggregate_windows(emb)  # (1, T/w, w*sigma)
        if agg.shape[1] == 0:
            # 防御极短输入（长度小于一个聚合窗口）。
            return torch.empty(0, device=x_tf.device, dtype=x_tf.dtype)
        out, _ = self.lstm(agg)
        logits = self.classifier(out)
        if logits.dim() == 2:
            logits = logits.squeeze(0)
        else:
            logits = logits.squeeze(-1).squeeze(0)
        return logits


def local_maxima(probs: torch.Tensor | np.ndarray, filter_size: int = 9, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    input_is_numpy = isinstance(probs, np.ndarray)
    tensor = torch.as_tensor(probs) if input_is_numpy else probs

    assert len(tensor.shape) in (1, 2), "Input tensor should have 1 or 2 dimensions"
    assert filter_size % 2 == 1, "Filter size should be an odd number"

    original_shape = tensor.shape
    if len(original_shape) == 1:
        tensor = tensor.unsqueeze(0)  # (1, T)

    padding = filter_size // 2
    padded_arr = F.pad(tensor, (padding, padding), mode="constant", value=-torch.inf)
    rolling_view = padded_arr.unfold(1, filter_size, step)

    center = filter_size // 2
    local_maxima_mask = torch.eq(rolling_view[:, :, center], torch.max(rolling_view, dim=-1).values)
    target_tensor = local_maxima_mask
    if step != 1:
        target_tensor = torch.zeros(
            (local_maxima_mask.shape[0], local_maxima_mask.shape[1] * step),
            dtype=torch.bool,
            device=local_maxima_mask.device,
        )
        for i in range(local_maxima_mask.shape[1]):
            target_tensor[:, i * step] = local_maxima_mask[:, i]
        target_tensor = target_tensor[:, : tensor.shape[1]]

    output_arr = torch.zeros_like(tensor)
    output_arr[target_tensor] = tensor[target_tensor]
    output_arr = output_arr.reshape(original_shape)

    if input_is_numpy:
        return output_arr.detach().cpu().numpy(), []
    return output_arr, []


def tensor_to_time(pred_mask: torch.Tensor | np.ndarray, sr: int, hop_length: int) -> list[float]:
    if isinstance(pred_mask, torch.Tensor):
        mask = pred_mask.detach().cpu().numpy().astype(bool).reshape(-1)
    else:
        mask = np.asarray(pred_mask).astype(bool).reshape(-1)
    idx = np.where(mask)[0]
    return [float(i * hop_length / sr) for i in idx]


def process_prob_sections(
    probs: torch.Tensor,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    loss_func: nn.Module,
    sr: int,
    hop_length: int,
    filter_size: int = 9,
    step: int = 1,
    threshold: float = 0.05,
) -> list[float]:
    del outputs, labels, loss_func  # kept for API compatibility with author code
    prob_sections, _ = local_maxima(probs, filter_size=filter_size, step=step)
    _valid_probs = prob_sections[prob_sections > 0]

    # 使用调用方传入的阈值（移除硬编码覆盖）。
    pred_mask = prob_sections >= threshold
    pred_times = tensor_to_time(pred_mask, sr=sr, hop_length=hop_length)
    return pred_times


def match_predictions(pred_times: list[float], true_times: list[float], tolerance: float = 0.5) -> tuple[float, float, float]:
    matched = 0
    used: set[int] = set()
    for t_pred in pred_times:
        for i, t_true in enumerate(true_times):
            if i in used:
                continue
            if abs(t_pred - t_true) <= tolerance:
                matched += 1
                used.add(i)
                break
    precision = matched / len(pred_times) if pred_times else 0.0
    recall = matched / len(true_times) if true_times else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_boundary_match_details(pred_times: list[float], true_times: list[float]) -> dict[str, Any]:
    """构建逐边界匹配详情（用于分析预测与标注的时间差）。"""
    pred = [float(t) for t in pred_times]
    true = [float(t) for t in true_times]
    matched: list[dict[str, float | bool]] = []

    if pred and true:
        all_pairs: list[tuple[float, int, int]] = []
        for i, p in enumerate(pred):
            for j, t in enumerate(true):
                all_pairs.append((abs(p - t), i, j))
        all_pairs.sort(key=lambda x: x[0])

        used_pred: set[int] = set()
        used_true: set[int] = set()
        for _abs_err, i, j in all_pairs:
            if i in used_pred or j in used_true:
                continue
            used_pred.add(i)
            used_true.add(j)
            err = pred[i] - true[j]
            matched.append(
                {
                    "pred_time": float(pred[i]),
                    "true_time": float(true[j]),
                    "error_sec": float(err),
                    "abs_error_sec": float(abs(err)),
                    "within_0.5s": bool(abs(err) <= 0.5),
                    "within_3s": bool(abs(err) <= 3.0),
                }
            )

        unmatched_pred = [float(pred[i]) for i in range(len(pred)) if i not in used_pred]
        unmatched_true = [float(true[j]) for j in range(len(true)) if j not in used_true]
    else:
        unmatched_pred = [float(x) for x in pred]
        unmatched_true = [float(x) for x in true]

    abs_errs = [float(x["abs_error_sec"]) for x in matched]
    mae = float(np.mean(abs_errs)) if abs_errs else None
    medae = float(np.median(abs_errs)) if abs_errs else None
    maxae = float(np.max(abs_errs)) if abs_errs else None

    return {
        "matched_pairs": matched,
        "matched_count": len(matched),
        "unmatched_pred_times": unmatched_pred,
        "unmatched_true_times": unmatched_true,
        "false_positive_count": len(unmatched_pred),
        "false_negative_count": len(unmatched_true),
        "mae_abs_error_sec": mae,
        "median_abs_error_sec": medae,
        "max_abs_error_sec": maxae,
    }


def _fmt_list_for_csv(values: list[float]) -> str:
    return ";".join(f"{float(v):.3f}" for v in values)


def save_boundary_diff_files(run_dir: Path, split_name: str, song_details: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = run_dir / f"boundary_diff_{split_name}.json"
    csv_path = run_dir / f"boundary_diff_{split_name}.csv"

    # JSON 全量保存，便于后续二次分析/画图。
    save_json(
        json_path,
        {
            "split": split_name,
            "num_songs": len(song_details),
            "songs": song_details,
        },
    )

    # CSV 便于快速浏览与筛选。
    fieldnames = [
        "song_id",
        "title",
        "filename",
        "duration_sec",
        "num_true",
        "num_pred",
        "matched_count",
        "false_positive_count",
        "false_negative_count",
        "mae_abs_error_sec",
        "median_abs_error_sec",
        "max_abs_error_sec",
        "true_times",
        "pred_times",
        "unmatched_true_times",
        "unmatched_pred_times",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in song_details:
            writer.writerow(
                {
                    "song_id": row.get("song_id"),
                    "title": row.get("title"),
                    "filename": row.get("filename"),
                    "duration_sec": row.get("duration_sec"),
                    "num_true": row.get("num_true"),
                    "num_pred": row.get("num_pred"),
                    "matched_count": row.get("matched_count"),
                    "false_positive_count": row.get("false_positive_count"),
                    "false_negative_count": row.get("false_negative_count"),
                    "mae_abs_error_sec": row.get("mae_abs_error_sec"),
                    "median_abs_error_sec": row.get("median_abs_error_sec"),
                    "max_abs_error_sec": row.get("max_abs_error_sec"),
                    "true_times": _fmt_list_for_csv(row.get("true_times", [])),
                    "pred_times": _fmt_list_for_csv(row.get("pred_times", [])),
                    "unmatched_true_times": _fmt_list_for_csv(row.get("unmatched_true_times", [])),
                    "unmatched_pred_times": _fmt_list_for_csv(row.get("unmatched_pred_times", [])),
                }
            )
    return json_path, csv_path


def match_predictions_segment(pre_section: list[float], true_times: list[float], tolerance: float = 3.0) -> tuple[float, float, float]:
    if len(true_times) < 2:
        return 0.0, 0.0, 0.0

    true_section = true_times
    true_section = [[true_section[i], true_section[i + 1]] for i in range(len(true_section) - 1)]

    if len(pre_section) > 1:
        pre_section = [[pre_section[i], pre_section[i + 1]] for i in range(len(pre_section) - 1)]
        try:
            import mir_eval

            detection_results = mir_eval.segment.detection(
                np.array(true_section),
                np.array(pre_section),
                window=tolerance,
                beta=1.0,
                trim=False,
            )
            precision = float(detection_results[0])
            recall = float(detection_results[1])
            f1 = float(detection_results[2])
        except ModuleNotFoundError:
            # 若 mir_eval 不可用，则退化为点级匹配计算。
            pred_boundary = [x[0] for x in pre_section]
            precision, recall, f1 = match_predictions(pred_boundary, true_times, tolerance=tolerance)
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    return precision, recall, f1


def build_segment_boundaries(boundary_times: list[float], duration_sec: float) -> list[float]:
    """将内部边界补全为完整边界列表 [0, ..., duration]。"""
    eps = 1e-6
    if duration_sec <= eps:
        return []
    mids = sorted(float(t) for t in boundary_times if eps < float(t) < (duration_sec - eps))
    out = [0.0] + mids + [float(duration_sec)]
    # 去重并保证严格单调递增。
    cleaned: list[float] = []
    for t in out:
        if not cleaned or (t - cleaned[-1]) > eps:
            cleaned.append(t)
    if len(cleaned) < 2:
        cleaned = [0.0, float(duration_sec)]
    return cleaned


def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
    epoch_idx: int,
    debug_batch_crash: bool = False,
    crash_dump_dir: Path | None = None,
) -> dict[str, float]:
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    steps = 0
    skipped = 0

    for step, batch in enumerate(loader, start=1):
        x = None
        y = None
        logits = None
        dbg_x_stats = tensor_stats(batch.get("x"))
        dbg_y_stats = tensor_stats(batch.get("y"))
        dbg_logits_stats: dict[str, Any] | None = None
        try:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            n = min(logits.numel(), y.numel())
            if n <= 0:
                skipped += 1
                continue
            logits = logits[:n]
            y = y[:n]
            dbg_logits_stats = tensor_stats(logits)
            loss = criterion(logits, y)
            (loss / grad_accum_steps).backward()

            if step % grad_accum_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            steps += 1
        except Exception as err:
            if debug_batch_crash:
                payload = {
                    "stage": "train",
                    "epoch": epoch_idx,
                    "step": step,
                    "error_type": type(err).__name__,
                    "error_message": str(err),
                    "song_id": batch.get("song_id"),
                    "title": batch.get("title"),
                    "filename": batch.get("filename"),
                    "boundary_times": batch.get("boundary_times"),
                    "x_stats": dbg_x_stats if dbg_x_stats is not None else tensor_stats(x),
                    "y_stats": dbg_y_stats if dbg_y_stats is not None else tensor_stats(y),
                    "logits_stats": dbg_logits_stats if dbg_logits_stats is not None else tensor_stats(logits),
                }
                if crash_dump_dir is not None:
                    crash_dump_path = crash_dump_dir / f"train_epoch{epoch_idx:03d}_step{step:04d}.json"
                    save_json(crash_dump_path, payload)
                    print(f"[CRASH-DEBUG] Saved diagnostics: {crash_dump_path}")
                else:
                    print(f"[CRASH-DEBUG] {payload}")
                if device.type == "cuda":
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        pass
            raise

    return {"loss": total_loss / max(1, steps), "num_songs": float(steps), "skipped_songs": float(skipped)}


@torch.no_grad()
def run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    w_seconds: float,
    hop_length: int,
    smooth_kernel: int,
    peak_threshold: float,
    sr: int,
    epoch_idx: int,
    debug_batch_crash: bool = False,
    crash_dump_dir: Path | None = None,
    collect_song_details: bool = False,
) -> dict[str, Any]:
    model.train(False)
    total_loss = 0.0
    steps = 0
    skipped = 0
    sum_p3 = sum_r3 = sum_f3 = 0.0
    sum_p05 = sum_r05 = sum_f05 = 0.0
    peak_count_total = 0
    max_prob_total = 0.0
    song_details: list[dict[str, Any]] = []

    for step, batch in enumerate(loader, start=1):
        x = None
        y = None
        logits = None
        dbg_x_stats = tensor_stats(batch.get("x"))
        dbg_y_stats = tensor_stats(batch.get("y"))
        dbg_logits_stats: dict[str, Any] | None = None
        try:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            n = min(logits.numel(), y.numel())
            if n <= 0:
                skipped += 1
                continue
            logits = logits[:n]
            y = y[:n]
            dbg_logits_stats = tensor_stats(logits)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
            steps += 1

            probs = torch.sigmoid(logits)
            # 输出位于聚合后的时间分辨率（每步约为 w_seconds），
            # 因此使用 effective_hop = fold_size * hop_length 将索引映射回时间。
            frame_duration = hop_length / sr
            fold_size = max(1, int(w_seconds / frame_duration))
            effective_hop = max(1, fold_size * hop_length)
            pred_times = process_prob_sections(
                probs,
                logits,
                y,
                criterion,
                sr=sr,
                hop_length=effective_hop,
                filter_size=smooth_kernel,
                step=1,
                threshold=peak_threshold,
            )

            probs_np = probs.detach().cpu().numpy().astype(np.float32).reshape(-1)
            peak_count_total += len(pred_times)
            max_prob_total += float(probs_np.max()) if probs_np.size > 0 else 0.0

            true_times_internal = sorted(float(t) for t in batch["boundary_times"])
            song_duration_sec = float(batch["x"].shape[0] * (hop_length / sr))

            p05, r05, f05 = match_predictions(pred_times, true_times_internal, tolerance=0.5)
            pre_for_seg = build_segment_boundaries(pred_times, song_duration_sec)
            true_for_seg = build_segment_boundaries(true_times_internal, song_duration_sec)
            p3, r3, f3 = match_predictions_segment(pre_for_seg, true_for_seg, tolerance=3.0)

            if collect_song_details:
                detail = build_boundary_match_details(pred_times, true_times_internal)
                song_details.append(
                    {
                        "song_id": batch.get("song_id"),
                        "title": batch.get("title"),
                        "filename": batch.get("filename"),
                        "duration_sec": float(song_duration_sec),
                        "num_true": len(true_times_internal),
                        "num_pred": len(pred_times),
                        "true_times": [float(t) for t in true_times_internal],
                        "pred_times": [float(t) for t in pred_times],
                        "p05": float(p05),
                        "r05": float(r05),
                        "f05": float(f05),
                        "p3": float(p3),
                        "r3": float(r3),
                        "f3": float(f3),
                        **detail,
                    }
                )

            sum_p3 += p3
            sum_r3 += r3
            sum_f3 += f3
            sum_p05 += p05
            sum_r05 += r05
            sum_f05 += f05
        except Exception as err:
            if debug_batch_crash:
                payload = {
                    "stage": "eval",
                    "epoch": epoch_idx,
                    "step": step,
                    "error_type": type(err).__name__,
                    "error_message": str(err),
                    "song_id": batch.get("song_id"),
                    "title": batch.get("title"),
                    "filename": batch.get("filename"),
                    "boundary_times": batch.get("boundary_times"),
                    "x_stats": dbg_x_stats if dbg_x_stats is not None else tensor_stats(x),
                    "y_stats": dbg_y_stats if dbg_y_stats is not None else tensor_stats(y),
                    "logits_stats": dbg_logits_stats if dbg_logits_stats is not None else tensor_stats(logits),
                }
                if crash_dump_dir is not None:
                    crash_dump_path = crash_dump_dir / f"eval_epoch{epoch_idx:03d}_step{step:04d}.json"
                    save_json(crash_dump_path, payload)
                    print(f"[CRASH-DEBUG] Saved diagnostics: {crash_dump_path}")
                else:
                    print(f"[CRASH-DEBUG] {payload}")
                if device.type == "cuda":
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        pass
            raise

    p3 = sum_p3 / max(1, steps)
    r3 = sum_r3 / max(1, steps)
    f3 = sum_f3 / max(1, steps)
    p05 = sum_p05 / max(1, steps)
    r05 = sum_r05 / max(1, steps)
    f05 = sum_f05 / max(1, steps)
    avg_pred_peaks = peak_count_total / max(1, steps)
    avg_max_prob = max_prob_total / max(1, steps)
    out = {
        "loss": total_loss / max(1, steps),
        "num_songs": float(steps),
        "skipped_songs": float(skipped),
        "avg_pred_peaks": avg_pred_peaks,
        "avg_max_prob": avg_max_prob,
        "hr3p": p3,
        "hr3r": r3,
        "hr3f": f3,
        "hr05p": p05,
        "hr05r": r05,
        "hr05f": f05,
    }
    if collect_song_details:
        out["song_details"] = song_details
    return out


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 for this task")
    if args.w_seconds <= 0:
        raise ValueError("--w-seconds must be > 0")

    root = Path(args.root).resolve()
    dataset_json = (root / args.dataset_json).resolve()
    feature_dir = (root / args.feature_dir).resolve()
    feature_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir).resolve() if args.run_dir else (root / "runs" / f"sa_cnfolk_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.benchmark = True
    device = pick_device(args.device)

    records = load_records(dataset_json)
    maybe_extract_mel(records, root=root, feature_dir=feature_dir, args=args)
    validate_features(records, feature_dir=feature_dir, n_mels=args.n_mels)
    splits, split_meta = split_by_title(records, seed=args.seed)

    train_ds = MelBoundaryDataset(
        records=splits["train"],
        feature_dir=feature_dir,
        sr=args.sr,
        hop_length=args.hop_length,
        tolerance=args.tolerance,
        w_seconds=args.w_seconds,
    )
    val_ds = MelBoundaryDataset(
        records=splits["val"],
        feature_dir=feature_dir,
        sr=args.sr,
        hop_length=args.hop_length,
        tolerance=args.tolerance,
        w_seconds=args.w_seconds,
    )
    test_ds = MelBoundaryDataset(
        records=splits["test"],
        feature_dir=feature_dir,
        sr=args.sr,
        hop_length=args.hop_length,
        tolerance=args.tolerance,
        w_seconds=args.w_seconds,
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch_size_one)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch_size_one)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch_size_one)

    model = SACNFolkModel(
        n_mels=args.n_mels,
        sigma=args.sigma,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        w_seconds=args.w_seconds,
        sr=args.sr,
        hop_length=args.hop_length,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 兼容旧版 torch：某些版本的 ReduceLROnPlateau 不支持 verbose 参数。
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=args.lr_plateau_patience,
            factor=args.lr_plateau_factor,
            verbose=True,
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=args.lr_plateau_patience,
            factor=args.lr_plateau_factor,
        )

    split_summary = {
        "n_total": len(records),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_used": len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
        "n_dropped": len(records) - (len(splits["train"]) + len(splits["val"]) + len(splits["test"])),
        "n_titles_total": len({r["title"] for r in records}),
        "n_titles_train": len({r["title"] for r in splits["train"]}),
        "n_titles_val": len({r["title"] for r in splits["val"]}),
        "n_titles_test": len({r["title"] for r in splits["test"]}),
        "split_meta": split_meta,
    }
    save_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "device": str(device),
            "split_summary": split_summary,
        },
    )

    best_epoch = 0
    best_hr3f = -1.0
    best_val_loss = 1e18
    no_improve = 0
    history: list[dict[str, Any]] = []

    print("=== SA-CNFolk Training Start ===")
    print(f"device: {device}")
    print(f"run_dir: {run_dir}")
    print(f"cudnn_enabled: {torch.backends.cudnn.enabled}")
    print(f"debug_batch_crash: {args.debug_batch_crash}")
    print(f"grad_accum_steps: {max(1, args.grad_accum_steps)}")
    print(
        f"split: train={split_summary['n_train']} "
        f"val={split_summary['n_val']} "
        f"test={split_summary['n_test']} "
        f"(used={split_summary['n_used']}, dropped={split_summary['n_dropped']})"
    )
    print(
        "split_titles: "
        f"train={split_summary['n_titles_train']} "
        f"val={split_summary['n_titles_val']} "
        f"test={split_summary['n_titles_test']} "
        f"(total={split_summary['n_titles_total']})"
    )
    print(
        "split_regions: "
        f"used={split_meta['regions_used']} "
        f"skipped_lt4={split_meta['regions_skipped_lt4']} "
        f"total={split_meta['regions_total']}"
    )
    crash_dump_dir = run_dir / "crash_debug" if args.debug_batch_crash else None
    if crash_dump_dir is not None:
        crash_dump_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch_train(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=max(1, args.grad_accum_steps),
            epoch_idx=epoch,
            debug_batch_crash=args.debug_batch_crash,
            crash_dump_dir=crash_dump_dir,
        )
        val_metrics = run_epoch_eval(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            w_seconds=args.w_seconds,
            hop_length=args.hop_length,
            smooth_kernel=args.smooth_kernel,
            peak_threshold=args.peak_threshold,
            sr=args.sr,
            epoch_idx=epoch,
            debug_batch_crash=args.debug_batch_crash,
            crash_dump_dir=crash_dump_dir,
        )
        epoch_seconds = time.perf_counter() - epoch_t0

        row = {
            "epoch": epoch,
            "lr": lr,
            "epoch_seconds": epoch_seconds,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        # 与作者代码保持一致：scheduler.step(avg_f1_val)
        scheduler.step(float(val_metrics["hr3f"]))

        print(
            f"[Epoch {epoch:03d}] "
            f"lr={lr:.6g} "
            f"train_songs={int(train_metrics['num_songs'])} "
            f"train_skipped={int(train_metrics['skipped_songs'])} "
            f"val_songs={int(val_metrics['num_songs'])} "
            f"val_skipped={int(val_metrics['skipped_songs'])} "
            f"epoch_time={epoch_seconds:.2f}s | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"avg_peaks={val_metrics['avg_pred_peaks']:.2f} "
            f"avg_max_prob={val_metrics['avg_max_prob']:.3f} "
            f"HR3F={val_metrics['hr3f']:.4f} "
            f"HR.5F={val_metrics['hr05f']:.4f}"
        )

        improved = False
        if val_metrics["hr3f"] > best_hr3f + 1e-8:
            improved = True
        elif abs(val_metrics["hr3f"] - best_hr3f) <= 1e-8 and val_metrics["loss"] < best_val_loss:
            improved = True

        if improved:
            best_epoch = epoch
            best_hr3f = float(val_metrics["hr3f"])
            best_val_loss = float(val_metrics["loss"])
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_hr3f": best_hr3f,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                    "split_summary": split_summary,
                },
                run_dir / "best.pt",
            )
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}, patience={args.early_stop_patience}")
            break

    save_json(run_dir / "history.json", {"history": history, "best_epoch": best_epoch, "best_hr3f": best_hr3f})

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = run_epoch_eval(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        w_seconds=args.w_seconds,
        hop_length=args.hop_length,
        smooth_kernel=args.smooth_kernel,
        peak_threshold=args.peak_threshold,
        sr=args.sr,
        epoch_idx=best_epoch,
        debug_batch_crash=args.debug_batch_crash,
        crash_dump_dir=crash_dump_dir,
        collect_song_details=args.export_boundary_diff,
    )
    test_song_details = test_metrics.pop("song_details", None)
    save_json(
        run_dir / "final_report.json",
        {"best_epoch": best_epoch, "best_hr3f": best_hr3f, "best_val_loss": best_val_loss, "test_metrics": test_metrics},
    )
    if test_song_details is not None:
        json_path, csv_path = save_boundary_diff_files(
            run_dir=run_dir,
            split_name="test",
            song_details=test_song_details,
        )
        print(f"boundary_diff_json: {json_path}")
        print(f"boundary_diff_csv: {csv_path}")

    print("=== SA-CNFolk Training Done ===")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_hr3f: {best_hr3f:.4f}")
    print(
        f"test HR3F={test_metrics['hr3f']:.4f}, "
        f"HR.5F={test_metrics['hr05f']:.4f}, "
        f"loss={test_metrics['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
