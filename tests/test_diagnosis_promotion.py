"""Synthetic count fixtures for candidate selection/review, not model performance."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnosis_evaluation import classification_metrics, promotion_gate, select_candidate


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


def test_selection_rejects_false_alarm_hidden_by_rounding():
    over_limit = count_report([[9_499_996, 500_004], [0, 10_000_000]])
    within_limit = count_report([[9_500_000, 500_000], [500_000, 9_500_000]])
    assert over_limit["normal_false_alarm_rate"] == within_limit["normal_false_alarm_rate"] == .05
    assert over_limit["macro_f1"] > within_limit["macro_f1"]
    reports = {"over_limit": over_limit, "within_limit": within_limit}
    original = deepcopy(reports)
    result = select_candidate(reports)
    assert result["selected_candidate"] == "within_limit"
    assert result["feasible_candidates"] == result["ranked_candidates"] == ["within_limit"]
    assert result["constraint_met"]
    assert result["selection_pool"] == "feasible_candidates"
    assert result["comparison_basis"] == "confusion_matrix_counts"
    assert result["ranking_metric"] == "macro_f1"
    assert reports == original
    json.dumps(result, allow_nan=False)


def test_selection_distinguishes_macro_f1_hidden_by_rounding():
    lower = count_report([[9_099_996, 900_004], [900_004, 9_099_996]])
    higher = count_report([[9_100_000, 900_000], [900_000, 9_100_000]])
    assert lower["macro_f1"] == higher["macro_f1"] == .91
    result = select_candidate({"lower_first": lower, "higher_second": higher}, max_false_alarm=.1)
    assert result["ranked_candidates"] == ["higher_second", "lower_first"]
    assert result["selected_candidate"] == "higher_second"


def test_selection_preserves_search_order_only_on_exact_ties(reports):
    report = reports[1]
    for ids in (("second", "first"), ("first", "second")):
        result = select_candidate(dict.fromkeys(ids, report))
        assert result["selected_candidate"] == ids[0]
        assert result["candidate_order"] == result["ranked_candidates"] == list(ids)
        assert result["tie_break"] == "first_in_search_order"


def test_selection_keeps_macro_f1_priority_with_rare_faults():
    majority_only = count_report([[1000, 0, 0], [10, 0, 0], [2, 0, 0]])
    rare_faults = count_report([[960, 40, 0], [2, 8, 0], [1, 0, 1]])
    assert majority_only["accuracy"] > rare_faults["accuracy"]
    assert majority_only["macro_f1"] < rare_faults["macro_f1"]
    result = select_candidate({"majority_only": majority_only, "rare_faults": rare_faults})
    assert result["selected_candidate"] == "rare_faults"
    assert result["constraint_met"]


def test_selection_fallback_is_explicit_and_not_a_promotion():
    lower = count_report([[9_099_996, 900_004], [900_004, 9_099_996]])
    higher = count_report([[9_100_000, 900_000], [900_000, 9_100_000]])
    result = select_candidate({"lower": lower, "higher": higher})
    assert not result["constraint_met"]
    assert result["feasible_candidates"] == []
    assert result["selection_pool"] == "unconstrained_fallback"
    assert result["selected_candidate"] == "higher"
    assert result["ranked_candidates"] == ["higher", "lower"]
    assert "eligible" not in result


def test_selection_supports_normal_as_last_class():
    champion = count_report([[30, 20], [50, 900]], ["fault", "정상"])
    candidate = count_report([[40, 10], [10, 940]], ["fault", "정상"])
    result = select_candidate({"champion": champion, "candidate": candidate})
    assert result["feasible_candidates"] == ["candidate"]
    assert result["selected_candidate"] == "candidate"


@pytest.mark.parametrize("reports", [None, [], {}, {"": {}}, {" ": {}}, {0: {}}])
def test_selection_requires_named_reports(reports):
    with pytest.raises(ValueError, match="candidate IDs"):
        select_candidate(reports)


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1", "normal_false_alarm_rate", "fault_miss_rate"])
def test_selection_does_not_ignore_invalid_candidates(reports, metric):
    reports[0][metric] = float("nan")
    with pytest.raises(ValueError, match=f"invalid selection metrics for invalid: invalid_{metric}"):
        select_candidate({"valid": reports[1], "invalid": reports[0]})


@pytest.mark.parametrize("counts,names,field", [
    ([[98, 2], [2, 98]], ["fault-1", "정상"], "class_order"),
    ([[98, 2], [2, 99]], None, "rows"),
    ([[97, 2], [2, 99]], None, "support"),
])
def test_selection_refuses_different_holdout_populations(counts, names, field):
    baseline = count_report([[90, 10], [10, 90]])
    different = count_report(counts, names)
    with pytest.raises(ValueError, match=f"selection {field} mismatch for different"):
        select_candidate({"baseline": baseline, "different": different})


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), float("-inf"), -.1, 1.1, True, None, "0.05"])
def test_selection_rejects_invalid_limits(reports, limit):
    with pytest.raises(ValueError, match="selection false-alarm limit"):
        select_candidate({"candidate": reports[1]}, max_false_alarm=limit)


@pytest.mark.parametrize("counts,limit", [([[100, 0], [0, 100]], 0), ([[0, 100], [0, 100]], 1)])
def test_selection_allows_inclusive_constraint_extremes(counts, limit):
    assert select_candidate({"candidate": count_report(counts)}, max_false_alarm=limit)["constraint_met"]
