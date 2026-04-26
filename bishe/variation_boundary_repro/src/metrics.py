from __future__ import annotations


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
    }

