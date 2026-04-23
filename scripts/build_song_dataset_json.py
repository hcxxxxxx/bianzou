#!/usr/bin/env python3
"""Build training JSON from dataset_file.xlsx, area.xlsx and structure.xlsx.

Usage example:
    python3 scripts/build_song_dataset_json.py \
      --root . \
      --dataset dataset_file.xlsx \
      --area area.xlsx \
      --structure structure.xlsx \
      --audio-dir "民歌变奏曲wav文件" \
      --output songs_dataset.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace(" ", "")
    text = text.replace("（", "(").replace("）", ")")
    return text


def parse_boundary_time(value: Any) -> float | None:
    """Parse minute.second-like notation into seconds.

    Examples:
    - 1.15 -> 75.0  (1 min 15 sec)
    - 1.5  -> 110.0 (1 min 50 sec)
    - 0.53 -> 53.0
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    text = text.replace("：", ":")

    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            minute_text, second_text = parts
            if not minute_text:
                minute_text = "0"
            if not second_text:
                second_text = "0"
            try:
                total = float(minute_text) * 60 + float(second_text)
                return round(total, 2)
            except ValueError:
                return None

    match = re.match(r"^(-?\d+)(?:\.(\d+))?$", text)
    if not match:
        return None

    minutes = int(match.group(1))
    second_digits = match.group(2) or ""

    if not second_digits:
        seconds = 0
    elif len(second_digits) == 1:
        seconds = int(second_digits) * 10
    else:
        seconds = int(second_digits[:2])

    total = minutes * 60 + seconds
    return round(float(total), 2)


def extract_start_columns(columns: list[str]) -> list[str]:
    indexed_columns: list[tuple[int, str]] = []
    for col in columns:
        col_text = str(col)
        lower_col = col_text.lower()
        if "变奏段" not in col_text:
            continue
        if "start" not in lower_col and "star" not in lower_col:
            continue
        match = re.search(r"变奏段\s*(\d+)", col_text)
        if not match:
            continue
        indexed_columns.append((int(match.group(1)), col_text))

    indexed_columns.sort(key=lambda item: item[0])
    return [name for _, name in indexed_columns]


def build_index(rows: pd.DataFrame, key_columns: list[str]) -> dict[str, list[dict[str, Any]]]:
    idx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in rows.iterrows():
        key = "||".join(normalize_text(row[col]) for col in key_columns)
        idx[key].append(row.to_dict())
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filtered song JSON for training.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--dataset", default="dataset_file.xlsx", help="Path to dataset_file.xlsx")
    parser.add_argument("--area", default="area.xlsx", help="Path to area.xlsx")
    parser.add_argument("--structure", default="structure.xlsx", help="Path to structure.xlsx")
    parser.add_argument("--audio-dir", default="民歌变奏曲wav文件", help="Directory containing wav files")
    parser.add_argument("--output", default="songs_dataset.json", help="Output JSON path (relative to root)")
    parser.add_argument(
        "--exclude-regions",
        default="红色歌曲,新民歌",
        help="Comma-separated region tags to exclude",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_path = (root / args.dataset).resolve()
    area_path = (root / args.area).resolve()
    structure_path = (root / args.structure).resolve()
    audio_dir = (root / args.audio_dir).resolve()
    output_path = (root / args.output).resolve()

    excluded_regions = {normalize_text(tag) for tag in args.exclude_regions.split(",") if tag.strip()}

    dataset_df = pd.read_excel(dataset_path)
    area_df = pd.read_excel(area_path)
    structure_df = pd.read_excel(structure_path)

    # Remove template/example rows.
    dataset_df = dataset_df[~dataset_df["曲目"].astype(str).str.startswith("例：", na=False)].copy()
    area_df = area_df[~area_df["曲目名称"].astype(str).str.startswith("例：", na=False)].copy()
    structure_df = structure_df[~structure_df["曲目名称"].astype(str).str.startswith("例：", na=False)].copy()

    # Build area lookup by (title, performer)
    area_idx = build_index(area_df, ["曲目名称", "演奏（唱）者"])
    region_lookup: dict[str, str] = {}
    region_conflicts: dict[str, list[str]] = {}
    for key, rows in area_idx.items():
        regions = [normalize_text(r.get("地区")) for r in rows if normalize_text(r.get("地区"))]
        if not regions:
            continue
        unique_regions = list(dict.fromkeys(regions))
        if len(unique_regions) == 1:
            region_lookup[key] = unique_regions[0]
        else:
            region_conflicts[key] = unique_regions

    structure_cols = list(structure_df.columns)
    start_columns = extract_start_columns(structure_cols)

    structure_idx_k4 = build_index(structure_df, ["曲目名称", "演奏（唱）者", "音乐平台", "专辑名称"])

    records: list[dict[str, Any]] = []
    stats = Counter()
    skipped_records: list[dict[str, Any]] = []
    fallback_records: list[dict[str, Any]] = []
    non_increasing_boundary_records: list[dict[str, Any]] = []

    for _, row in dataset_df.iterrows():
        stats["dataset_rows"] += 1

        title = normalize_text(row.get("曲目"))
        performer = normalize_text(row.get("演奏"))
        source = normalize_text(row.get("来源"))
        album = normalize_text(row.get("专辑"))
        filename = normalize_text(row.get("filename"))

        if not title or not performer or not filename:
            stats["skipped_missing_basic_info"] += 1
            skipped_records.append(
                {
                    "reason": "missing_basic_info",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                }
            )
            continue

        key2 = f"{title}||{performer}"
        key4 = f"{title}||{performer}||{source}||{album}"

        struct_row: dict[str, Any] | None = None
        rows_k4 = structure_idx_k4.get(key4, [])

        if len(rows_k4) == 1:
            struct_row = rows_k4[0]
            stats["matched_structure_k4"] += 1
        elif len(rows_k4) > 1:
            stats["skipped_structure_key4_ambiguous"] += 1
            skipped_records.append(
                {
                    "reason": "structure_key4_ambiguous",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                }
            )
            continue
        else:
            stats["skipped_structure_key4_unmatched"] += 1
            skipped_records.append(
                {
                    "reason": "structure_key4_unmatched",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                }
            )
            continue

        if key2 in region_conflicts:
            stats["skipped_area_region_ambiguous"] += 1
            skipped_records.append(
                {
                    "reason": "area_region_ambiguous",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                    "area_regions": region_conflicts[key2],
                }
            )
            continue

        region_area = region_lookup.get(key2, "")
        region_structure = normalize_text(struct_row.get("地区"))

        if not region_area:
            stats["skipped_missing_area_region"] += 1
            skipped_records.append(
                {
                    "reason": "missing_area_region",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                }
            )
            continue

        if not region_structure:
            stats["skipped_missing_structure_region"] += 1
            skipped_records.append(
                {
                    "reason": "missing_structure_region",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                }
            )
            continue

        if region_area != region_structure:
            stats["skipped_region_mismatch"] += 1
            skipped_records.append(
                {
                    "reason": "region_mismatch_between_area_and_structure",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                    "area_region": region_area,
                    "structure_region": region_structure,
                }
            )
            continue

        region = region_area

        if region in excluded_regions:
            stats["excluded_by_region"] += 1
            skipped_records.append(
                {
                    "reason": "excluded_by_region",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                    "region": region,
                }
            )
            continue

        audio_file = audio_dir / f"{filename}.wav"
        if not audio_file.exists():
            stats["skipped_missing_audio"] += 1
            skipped_records.append(
                {
                    "reason": "missing_audio_file",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                    "expected_audio_path": audio_file.as_posix(),
                }
            )
            continue

        time_points: list[float] = []
        for col in start_columns:
            sec = parse_boundary_time(struct_row.get(col))
            if sec is None:
                continue
            if sec < 0:
                continue
            time_points.append(round(sec, 2))

        # Keep order, remove exact duplicates.
        deduped_times: list[float] = []
        for t in time_points:
            if not deduped_times or abs(t - deduped_times[-1]) > 1e-9:
                deduped_times.append(t)

        # Keep only strictly increasing boundaries for training stability.
        cleaned_times: list[float] = []
        dropped_points: list[dict[str, Any]] = []
        for t in deduped_times:
            if t <= 0:
                stats["dropped_non_increasing_boundary_points"] += 1
                dropped_points.append({"value": t, "reason": "non_positive"})
                continue
            if not cleaned_times or t > cleaned_times[-1]:
                cleaned_times.append(t)
            else:
                stats["dropped_non_increasing_boundary_points"] += 1
                dropped_points.append(
                    {
                        "value": t,
                        "reason": "non_increasing",
                        "previous_kept_value": cleaned_times[-1],
                    }
                )

        if dropped_points:
            non_increasing_boundary_records.append(
                {
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                    "raw_boundary_times": [0.0] + deduped_times,
                    "cleaned_boundary_times": [0.0] + cleaned_times,
                    "dropped_points": dropped_points,
                }
            )
            stats["skipped_non_increasing_boundary_records"] += 1
            skipped_records.append(
                {
                    "reason": "non_increasing_boundary_times",
                    "title": title,
                    "performer": performer,
                    "source": source,
                    "album": album,
                    "filename": filename,
                    "raw_boundary_times": [0.0] + deduped_times,
                    "dropped_points": dropped_points,
                }
            )
            continue

        boundary_times = [0.0] + cleaned_times
        segments = [{"label": "A", "start": 0.0}]
        for idx, t in enumerate(cleaned_times, start=1):
            segments.append({"label": f"A{idx}", "start": t})

        rel_audio_path = audio_file.relative_to(root).as_posix()
        song_id = f"{len(records) + 1:04d}"

        records.append(
            {
                "song_id": song_id,
                "title": title,
                "performer": performer,
                "region": region,
                "source": source,
                "album": album,
                "filename": filename,
                "audio_path": rel_audio_path,
                "segments": segments,
                "boundary_times": boundary_times,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    report_path = output_path.with_name(f"{output_path.stem}_build_report.json")
    report_payload = {
        "stats": dict(stats),
        "excluded_regions": sorted(excluded_regions),
        "skipped_reason_counts": dict(Counter(item["reason"] for item in skipped_records)),
        "fallback_reason_counts": dict(Counter(item["reason"] for item in fallback_records)),
        "skipped_records": skipped_records,
        "fallback_records": fallback_records,
        "non_increasing_boundary_records": non_increasing_boundary_records,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    print("=== Build Summary ===")
    print(f"dataset rows: {stats['dataset_rows']}")
    print(f"excluded by region ({','.join(sorted(excluded_regions))}): {stats['excluded_by_region']}")
    print(f"matched structure by key4: {stats['matched_structure_k4']}")
    print(f"skipped (structure key4 unmatched): {stats['skipped_structure_key4_unmatched']}")
    print(f"skipped (structure key4 ambiguous): {stats['skipped_structure_key4_ambiguous']}")
    print(f"skipped (area region ambiguous): {stats['skipped_area_region_ambiguous']}")
    print(f"skipped (missing area region): {stats['skipped_missing_area_region']}")
    print(f"skipped (missing structure region): {stats['skipped_missing_structure_region']}")
    print(f"skipped (area/structure region mismatch): {stats['skipped_region_mismatch']}")
    print(f"skipped (missing basic info): {stats['skipped_missing_basic_info']}")
    print(f"skipped (missing audio file): {stats['skipped_missing_audio']}")
    print(f"skipped (non-increasing boundaries): {stats['skipped_non_increasing_boundary_records']}")
    print(f"dropped non-increasing boundary points: {stats['dropped_non_increasing_boundary_points']}")
    print(f"region conflicts in area.xlsx: {len(region_conflicts)}")
    print(f"output songs: {len(records)}")
    print(f"output json: {output_path}")
    print("=== Skip Reasons ===")
    for reason, count in Counter(item["reason"] for item in skipped_records).most_common():
        print(f"{reason}: {count}")
    print("=== Fallback Reasons ===")
    for reason, count in Counter(item["reason"] for item in fallback_records).most_common():
        print(f"{reason}: {count}")
    print("=== Non-Increasing Boundary Records ===")
    print(f"records_with_dropped_non_increasing_points: {len(non_increasing_boundary_records)}")
    print(f"output report: {report_path}")


if __name__ == "__main__":
    main()
