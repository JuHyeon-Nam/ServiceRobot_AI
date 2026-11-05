"""Classification metrics, robot holdout, and candidate promotion criteria."""
from __future__ import annotations

from fractions import Fraction
from math import isfinite
from numbers import Integral, Real

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


def macro_f1_eval(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[str, float, bool]:
    """Unweighted sklearn-style LightGBM metric with a fixed multiclass denominator."""
    y = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or not len(y) or probabilities.ndim != 2
            or probabilities.shape[0] != len(y) or probabilities.shape[1] < 2):
        raise ValueError("macro-F1 requires labels (n,) and multiclass predictions (n, classes)")
    if (not np.isfinite(y).all() or not np.isfinite(probabilities).all()
            or (y != np.floor(y)).any() or (y < 0).any() or (y >= probabilities.shape[1]).any()):
        raise ValueError("macro-F1 requires finite predictions and valid integer class indices")
    score = f1_score(y.astype(int), probabilities.argmax(axis=1),
                     labels=np.arange(probabilities.shape[1]), average="macro", zero_division=0)
    return "macro_f1", float(score), True


def classification_metrics(y: np.ndarray, probabilities: np.ndarray, names: list[str]) -> dict:
    y, probabilities = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y), len(names)) or not len(y):
        raise ValueError("probability matrix must match rows and classes")
    if (y < 0).any() or (y >= len(names)).any() or not np.isfinite(probabilities).all():
        raise ValueError("invalid labels or probabilities")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("probabilities must sum to one")
    pred, labels = probabilities.argmax(axis=1), list(range(len(names)))
    normal = names.index("정상")
    normal_mask, fault_mask = y == normal, y != normal
    confidence, correct = probabilities.max(axis=1), pred == y
    ece, bins = 0.0, []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if mask.any():
            avg_confidence, accuracy = float(confidence[mask].mean()), float(correct[mask].mean())
            ece += float(mask.mean()) * abs(avg_confidence - accuracy)
            bins.append({"lower": float(lower), "upper": float(upper), "rows": int(mask.sum()),
                         "mean_confidence": round(avg_confidence, 6), "accuracy": round(accuracy, 6)})
    classes = classification_report(y, pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
    return {
        "rows": len(y), "accuracy": round(float(accuracy_score(y, pred)), 6),
        "macro_f1": round(float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)), 6),
        "normal_false_alarm_rate": round(float(np.mean(pred[normal_mask] != normal)), 6) if normal_mask.any() else None,
        "fault_miss_rate": round(float(np.mean(pred[fault_mask] == normal)), 6) if fault_mask.any() else None,
        "log_loss": round(float(log_loss(y, probabilities, labels=labels)), 6),
        "ece_10_bins": round(ece, 6), "calibration_bins": bins,
        "class_coverage": int(len(np.unique(y))),
        "per_class": {name: classes[name] for name in names},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(), "class_order": names,
    }


def asset_holdout(labels: np.ndarray, groups: np.ndarray, seed: int = 42,
                  test_size: float = 0.2, require_all_classes: bool = False) -> tuple[np.ndarray, np.ndarray]:
    labels, groups = np.asarray(labels), np.asarray(groups).astype(str)
    if len(labels) != len(groups) or any(not group.strip() for group in groups):
        raise ValueError("every row needs a nonempty asset ID")
    if len(np.unique(groups)) < 3:
        raise ValueError("diagnosis selection requires at least 3 independent assets")
    expected = set(labels.tolist())
    if require_all_classes:
        single_asset = [int(label) for label in expected if len(np.unique(groups[labels == label])) < 2]
        if single_asset:
            raise ValueError(f"classes need at least 2 independent assets for holdout: {single_asset}")
    for attempt in range(64):
        fit, selection = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                                random_state=seed + attempt).split(labels, groups=groups))
        coverage = set(labels[selection].tolist())
        selection_valid = coverage == expected if require_all_classes else len(coverage) > 1
        if set(labels[fit].tolist()) == expected and selection_valid:
            return fit, selection
    raise ValueError("holdout cannot preserve classes; more independent fault assets are needed")


def class_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    if mode not in ("unweighted", "sqrt_balanced", "balanced"):
        raise ValueError("unknown class weighting")
    labels = np.asarray(labels, dtype=int)
    if mode == "unweighted":
        return np.ones(len(labels))
    frequency = np.bincount(labels)
    weight = len(labels) / (len(np.unique(labels)) * frequency[labels])
    if mode == "sqrt_balanced":
        weight = np.sqrt(weight)
    weight = np.minimum(weight, 10.0)
    return weight / weight.mean()


def _unit_interval(value: object) -> bool:
    return (isinstance(value, Real) and not isinstance(value, bool)
            and 0 <= value <= 1 and isfinite(value))


def _promotion_evidence(report: dict) -> tuple[dict | None, list[str]]:
    """Validate reported scores against integer counts before policy comparisons."""
    if not isinstance(report, dict):
        return None, ["report_not_object"]
    metric_names = ("accuracy", "macro_f1", "normal_false_alarm_rate", "fault_miss_rate")
    errors = [f"invalid_{key}" for key in metric_names if not _unit_interval(report.get(key))]
    names = report.get("class_order")
    if (not isinstance(names, list) or len(names) < 2
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names) or "정상" not in names):
        return None, errors + ["invalid_class_order"]
    rows, coverage = report.get("rows"), report.get("class_coverage")
    if not isinstance(rows, Integral) or isinstance(rows, bool) or rows <= 0:
        errors.append("invalid_rows")
    if not isinstance(coverage, Integral) or isinstance(coverage, bool) or coverage != len(names):
        errors.append("incomplete_class_coverage")
    try:
        matrix = np.asarray(report.get("confusion_matrix"), dtype=object)
    except (TypeError, ValueError):
        return None, errors + ["invalid_confusion_matrix"]
    if matrix.shape != (len(names), len(names)) or any(
            not isinstance(value, Integral) or isinstance(value, bool) or value < 0 for value in matrix.flat):
        return None, errors + ["invalid_confusion_matrix"]

    # Python integers avoid overflow in count totals; fractions preserve threshold boundaries.
    counts = [[int(value) for value in row] for row in matrix]
    support = tuple(sum(row) for row in counts)
    predicted = [sum(column) for column in zip(*counts)]
    total = sum(support)
    if total != rows:
        errors.append("confusion_rows_mismatch")
    if any(count == 0 for count in support):
        errors.append("missing_validation_classes")
    if errors:
        return None, errors
    normal = names.index("정상")
    rates = {
        "accuracy": Fraction(sum(counts[i][i] for i in range(len(names))), total),
        "macro_f1": sum(Fraction(2 * counts[i][i], support[i] + predicted[i])
                        for i in range(len(names))) / len(names),
        "normal_false_alarm_rate": Fraction(support[normal] - counts[normal][normal], support[normal]),
        "fault_miss_rate": Fraction(sum(counts[i][normal] for i in range(len(names)) if i != normal),
                                    total - support[normal]),
    }
    # classification_metrics serializes these four scores to six decimal places.
    errors = [f"inconsistent_{key}" for key in metric_names
              if abs(float(report[key]) - float(rates[key])) > 5.000001e-7]
    if errors:
        return None, errors
    return {"rates": rates, "class_order": tuple(names), "support": support, "rows": total}, []


def promotion_gate(champion: dict, candidate: dict, max_false_alarm: float = 0.05,
                   min_macro_gain: float = 0.01, max_accuracy_drop: float = 0.005) -> dict:
    thresholds = {"min_macro_f1_gain": min_macro_gain, "max_accuracy_drop": max_accuracy_drop,
                  "max_normal_false_alarm": max_false_alarm}
    if any(not _unit_interval(value) for value in thresholds.values()):
        raise ValueError("promotion thresholds must be finite numbers between zero and one")
    thresholds = {key: float(value) for key, value in thresholds.items()}
    champion_evidence, champion_errors = _promotion_evidence(champion)
    candidate_evidence, candidate_errors = _promotion_evidence(candidate)
    reasons = []
    result = {"eligible": False, "reasons": reasons, "automatic_promotion": False,
              "comparison_basis": "confusion_matrix_counts", "thresholds": thresholds,
              "validation_errors": {"champion": champion_errors, "candidate": candidate_errors}}
    if champion_errors:
        reasons.append("invalid_champion_report")
    if candidate_errors:
        reasons.append("invalid_candidate_report")
    if reasons:
        return result
    for field in ("class_order", "rows", "support"):
        if champion_evidence[field] != candidate_evidence[field]:
            reasons.append(f"validation_{field}_mismatch")
    if reasons:
        return result
    before, after = champion_evidence["rates"], candidate_evidence["rates"]
    if after["macro_f1"] - before["macro_f1"] < Fraction(str(thresholds["min_macro_f1_gain"])):
        reasons.append("macro_f1_gain_below_threshold")
    if before["accuracy"] - after["accuracy"] > Fraction(str(thresholds["max_accuracy_drop"])):
        reasons.append("accuracy_regression")
    if after["normal_false_alarm_rate"] > Fraction(str(thresholds["max_normal_false_alarm"])):
        reasons.append("normal_false_alarm_limit")
    if after["fault_miss_rate"] > before["fault_miss_rate"]:
        reasons.append("fault_miss_regression")
    result["eligible"] = not reasons
    return result
