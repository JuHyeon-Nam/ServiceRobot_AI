"""Classification metrics, robot holdout, and candidate promotion criteria."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


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
                  test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    labels, groups = np.asarray(labels), np.asarray(groups).astype(str)
    if len(labels) != len(groups) or any(not group.strip() for group in groups):
        raise ValueError("every row needs a nonempty asset ID")
    if len(np.unique(groups)) < 3:
        raise ValueError("diagnosis selection requires at least 3 independent assets")
    expected = set(labels.tolist())
    for attempt in range(64):
        fit, selection = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                                random_state=seed + attempt).split(labels, groups=groups))
        if set(labels[fit].tolist()) == expected and len(np.unique(labels[selection])) > 1:
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


def promotion_gate(champion: dict, candidate: dict, max_false_alarm: float = 0.05,
                   min_macro_gain: float = 0.01, max_accuracy_drop: float = 0.005) -> dict:
    reasons = []
    if candidate["macro_f1"] < champion["macro_f1"] + min_macro_gain:
        reasons.append("macro_f1_gain_below_threshold")
    if candidate["accuracy"] < champion["accuracy"] - max_accuracy_drop:
        reasons.append("accuracy_regression")
    if candidate["normal_false_alarm_rate"] is None or candidate["normal_false_alarm_rate"] > max_false_alarm:
        reasons.append("normal_false_alarm_limit")
    if candidate["fault_miss_rate"] is None or champion["fault_miss_rate"] is None:
        reasons.append("missing_fault_observations")
    elif candidate["fault_miss_rate"] > champion["fault_miss_rate"]:
        reasons.append("fault_miss_regression")
    if candidate["class_coverage"] != len(candidate["class_order"]):
        reasons.append("missing_validation_classes")
    return {"eligible": not reasons, "reasons": reasons, "automatic_promotion": False,
            "thresholds": {"min_macro_f1_gain": min_macro_gain,
                           "max_accuracy_drop": max_accuracy_drop, "max_normal_false_alarm": max_false_alarm}}
