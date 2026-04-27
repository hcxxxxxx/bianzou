from __future__ import annotations

import numpy as np


def match_predictions(pred_times: list[float], true_times: list[float], tolerance: float) -> tuple[int, int, int]:
    matched = 0
    used: set[int] = set()
    for t_pred in sorted(pred_times):
        best_i = None
        best_dist = tolerance + 1e-9
        for i, t_true in enumerate(true_times):
            if i in used:
                continue
            dist = abs(t_pred - t_true)
            if dist <= tolerance and dist < best_dist:
                best_i = i
                best_dist = dist
        if best_i is not None:
            matched += 1
            used.add(best_i)
    return matched, len(pred_times), len(true_times)


def prf(matched: int, predicted: int, truth: int) -> dict[str, float]:
    precision = matched / predicted if predicted else 0.0
    recall = matched / truth if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_at_tolerance(rows: list[dict], tolerance: float) -> dict[str, float]:
    matched = 0
    predicted = 0
    truth = 0
    for row in rows:
        m, p, t = match_predictions(row["pred_times"], row["true_times"], tolerance)
        matched += m
        predicted += p
        truth += t
    result = prf(matched, predicted, truth)
    result.update({"matched": matched, "predicted": predicted, "truth": truth})
    return result


def evaluate_boundary_predictions(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        "HR.5": evaluate_at_tolerance(rows, tolerance=0.5),
        "HR3": evaluate_at_tolerance(rows, tolerance=3.0),
        "SEG3": evaluate_segment_predictions(rows, tolerance=3.0),
    }


def segment_intervals(boundary_times: list[float]) -> list[list[float]]:
    times = sorted(float(t) for t in boundary_times)
    return [[times[i], times[i + 1]] for i in range(len(times) - 1)]


def fallback_segment_detection(
    pred_times: list[float],
    true_times: list[float],
    tolerance: float,
) -> dict[str, float]:
    true_boundaries = sorted(set(t for interval in segment_intervals(true_times) for t in interval))
    pred_boundaries = sorted(set(t for interval in segment_intervals(pred_times) for t in interval))
    matched, predicted, truth = match_predictions(pred_boundaries, true_boundaries, tolerance)
    result = prf(matched, predicted, truth)
    result.update({"matched": matched, "predicted": predicted, "truth": truth})
    return result


def segment_detection(
    pred_times: list[float],
    true_times: list[float],
    tolerance: float = 3.0,
) -> dict[str, float]:
    true_sections = segment_intervals(true_times)
    pred_sections = segment_intervals(pred_times)
    if len(pred_sections) == 0 or len(true_sections) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched": 0,
            "predicted": len(pred_sections),
            "truth": len(true_sections),
        }

    try:
        import mir_eval.segment

        precision, recall, f1 = mir_eval.segment.detection(
            np.asarray(true_sections),
            np.asarray(pred_sections),
            window=tolerance,
            beta=1.0,
            trim=False,
        )
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "matched": 0,
            "predicted": len(pred_sections),
            "truth": len(true_sections),
        }
    except Exception:
        return fallback_segment_detection(pred_times, true_times, tolerance)


def evaluate_segment_predictions(rows: list[dict], tolerance: float = 3.0) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []
    predicted = 0
    truth = 0
    for row in rows:
        result = segment_detection(row["pred_times"], row["true_times"], tolerance=tolerance)
        precisions.append(result["precision"])
        recalls.append(result["recall"])
        f1s.append(result["f1"])
        predicted += int(result.get("predicted", 0))
        truth += int(result.get("truth", 0))
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "predicted": 0, "truth": 0}
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "predicted": predicted,
        "truth": truth,
    }
