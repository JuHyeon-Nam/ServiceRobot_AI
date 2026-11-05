"""Synthetic count fixtures for fail-closed candidate review, not model performance."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnosis_evaluation import classification_metrics, promotion_gate


def count_report(counts, names=None):
    matrix = np.asarray(counts, dtype=np.int64)
    classes = len(matrix)
    names = names or ["정상", *[f"fault-{i}" for i in range(1, classes)]]
    actual, predicted = np.repeat(np.arange(classes), classes), np.tile(np.arange(classes), classes)
    weights = matrix.ravel()
    normal = names.index("정상")
    support = matrix.sum(axis=1)
    return {
        "rows": int(matrix.sum()), "class_order": names, "class_coverage": int((support > 0).sum()),
        "confusion_matrix": matrix.tolist(),
        "accuracy": round(float(accuracy_score(actual, predicted, sample_weight=weights)), 6),
        "macro_f1": round(float(f1_score(actual, predicted, labels=np.arange(classes),
                                         average="macro", sample_weight=weights, zero_division=0)), 6),
        "normal_false_alarm_rate": round(float((support[normal] - matrix[normal, normal]) / support[normal]), 6),
        "fault_miss_rate": round(float((matrix[:, normal].sum() - matrix[normal, normal])
                                      / (matrix.sum() - support[normal])), 6),
    }


@pytest.fixture
def reports():
    return count_report([[900, 50], [20, 30]]), count_report([[940, 10], [10, 40]])


def test_valid_reports_are_eligible_but_never_automatically_promoted(reports):
    original = deepcopy(reports)
    result = promotion_gate(*reports)
    assert result["eligible"] and not result["automatic_promotion"]
    assert result["reasons"] == []
    assert result["validation_errors"] == {"champion": [], "candidate": []}
    assert result["comparison_basis"] == "confusion_matrix_counts"
    assert reports == original
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("side", ["champion", "candidate"])
@pytest.mark.parametrize("metric", ["accuracy", "macro_f1", "normal_false_alarm_rate", "fault_miss_rate"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None, -.1, 1.1, True, "0.9"])
def test_invalid_scores_fail_closed(reports, side, metric, value):
    reports[side == "candidate"][metric] = value
    result = promotion_gate(*reports)
    assert not result["eligible"]
    assert f"invalid_{side}_report" in result["reasons"]
    assert f"invalid_{metric}" in result["validation_errors"][side]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1", "normal_false_alarm_rate", "fault_miss_rate"])
def test_missing_or_inconsistent_scores_are_rejected(reports, metric):
    reports[1].pop(metric)
    assert f"invalid_{metric}" in promotion_gate(*reports)["validation_errors"]["candidate"]
    reports[1][metric] = .5
    result = promotion_gate(*reports)
    assert not result["eligible"]
    assert f"inconsistent_{metric}" in result["validation_errors"]["candidate"]


@pytest.mark.parametrize("field,value,error", [
    ("class_order", ["정상", "정상"], "invalid_class_order"),
    ("class_order", ["fault-a", "fault-b"], "invalid_class_order"),
    ("class_order", ["정상", " "], "invalid_class_order"),
    ("class_order", ["정상", []], "invalid_class_order"),
    ("class_order", "정상,fault", "invalid_class_order"),
    ("rows", None, "invalid_rows"),
    ("rows", True, "invalid_rows"),
    ("rows", 1000.0, "invalid_rows"),
    ("rows", -1, "invalid_rows"),
    ("rows", 999, "confusion_rows_mismatch"),
    ("class_coverage", 1, "incomplete_class_coverage"),
    ("class_coverage", 2.0, "incomplete_class_coverage"),
    ("confusion_matrix", None, "invalid_confusion_matrix"),
    ("confusion_matrix", [[940, 10], [10]], "invalid_confusion_matrix"),
    ("confusion_matrix", [[940, 10, 0], [10, 40, 0]], "invalid_confusion_matrix"),
    ("confusion_matrix", [[940., 10.], [10., 40.]], "invalid_confusion_matrix"),
    ("confusion_matrix", [[940, True], [10, 40]], "invalid_confusion_matrix"),
    ("confusion_matrix", [[940, -1], [10, 40]], "invalid_confusion_matrix"),
    ("confusion_matrix", [[940, "10"], [10, 40]], "invalid_confusion_matrix"),
    ("confusion_matrix", [[940, 60], [0, 0]], "missing_validation_classes"),
])
def test_invalid_evidence_is_rejected(reports, field, value, error):
    reports[1][field] = value
    result = promotion_gate(*reports)
    assert not result["eligible"]
    assert error in result["validation_errors"]["candidate"]


@pytest.mark.parametrize("value", [None, [], "report"])
def test_nonobject_reports_are_rejected(reports, value):
    result = promotion_gate(reports[0], value)
    assert not result["eligible"]
    assert result["validation_errors"]["candidate"] == ["report_not_object"]


def test_both_invalid_reports_are_audited(reports):
    for report in reports:
        report["macro_f1"] = float("nan")
    result = promotion_gate(*reports)
    assert result["reasons"] == ["invalid_champion_report", "invalid_candidate_report"]
    assert not result["eligible"]


def test_different_class_order_is_not_comparable():
    champion = count_report([[900, 100], [100, 900]])
    candidate = count_report([[980, 20], [20, 980]], ["fault-1", "정상"])
    result = promotion_gate(champion, candidate)
    assert result["reasons"] == ["validation_class_order_mismatch"]


@pytest.mark.parametrize("counts,reason", [
    ([[940, 10], [10, 41]], "validation_rows_mismatch"),
    ([[939, 10], [10, 41]], "validation_support_mismatch"),
])
def test_different_validation_populations_are_rejected(reports, counts, reason):
    result = promotion_gate(reports[0], count_report(counts))
    assert not result["eligible"]
    assert reason in result["reasons"]


def test_macro_gain_boundary_uses_counts_not_rounded_scores():
    champion = count_report([[9_000_000, 1_000_000], [1_000_000, 9_000_000]])
    exact = count_report([[9_100_000, 900_000], [900_000, 9_100_000]])
    rounded_up = count_report([[9_099_996, 900_004], [900_004, 9_099_996]])
    assert exact["macro_f1"] == rounded_up["macro_f1"] == .91
    assert promotion_gate(champion, exact, max_false_alarm=.1)["eligible"]
    result = promotion_gate(champion, rounded_up, max_false_alarm=.1)
    assert result["reasons"] == ["macro_f1_gain_below_threshold"]


def test_false_alarm_boundary_uses_counts_not_rounded_scores():
    champion = count_report([[9_400_000, 600_000], [1_000_000, 9_000_000]])
    exact = count_report([[9_500_000, 500_000], [500_000, 9_500_000]])
    rounded_down = count_report([[9_499_996, 500_004], [500_000, 9_500_000]])
    assert exact["normal_false_alarm_rate"] == rounded_down["normal_false_alarm_rate"] == .05
    assert promotion_gate(champion, exact)["eligible"]
    result = promotion_gate(champion, rounded_down)
    assert result["reasons"] == ["normal_false_alarm_limit"]


def test_accuracy_drop_boundary_is_inclusive():
    champion = count_report([[950, 0], [50, 0]])
    exact = count_report([[940, 10], [45, 5]])
    below = count_report([[939, 11], [45, 5]])
    assert promotion_gate(champion, exact)["eligible"]
    assert promotion_gate(champion, below)["reasons"] == ["accuracy_regression"]


def test_miss_regression_cannot_be_hidden_by_accuracy_or_macro_gain():
    champion = count_report([[100, 900], [0, 1000]])
    candidate = count_report([[1000, 0], [1, 999]])
    assert candidate["accuracy"] > champion["accuracy"]
    assert candidate["macro_f1"] > champion["macro_f1"]
    assert promotion_gate(champion, candidate)["reasons"] == ["fault_miss_regression"]


def test_all_nine_classes_and_normal_last_are_supported():
    classes = 9
    names = [*[f"fault-{i}" for i in range(classes - 1)], "정상"]
    actual = np.repeat(np.arange(classes), 100)
    reports = []
    for errors in (10, 1):
        predicted = actual.copy()
        for label in range(classes):
            predicted[label * 100:label * 100 + errors] = (label + 1) % classes
        reports.append(classification_metrics(actual, np.eye(classes)[predicted], names))
    assert promotion_gate(*reports)["eligible"]


def test_count_totals_do_not_overflow_int64(reports):
    for report in reports:
        scale = 2 ** 60
        report["confusion_matrix"] = [[value * scale for value in row] for row in report["confusion_matrix"]]
        report["rows"] *= scale
    assert promotion_gate(*reports)["eligible"]


@pytest.mark.parametrize("parameter", ["max_false_alarm", "min_macro_gain", "max_accuracy_drop"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1, True, "0.01"])
def test_invalid_policy_configuration_is_an_error(reports, parameter, value):
    with pytest.raises(ValueError, match="promotion thresholds"):
        promotion_gate(*reports, **{parameter: value})
