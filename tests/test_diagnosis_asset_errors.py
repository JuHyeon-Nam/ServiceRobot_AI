"""Robot-level error counts on synthetic labels, not measured robot performance."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnosis_evaluation import asset_error_analysis, classification_metrics

NAMES = ["fault-a", "정상", "fault-b", "absent-fault"]
Y = np.array([1, 1, 0, 0, 2, 2, 1, 0, 2])
PREDICTED = np.array([1, 0, 1, 2, 1, 2, 1, 0, 0])
GROUPS = np.array(["A", "A", "A", "A", "B", "B", "C", "D", "D"])


def test_robot_counts_reconstruct_the_complete_confusion_matrix_and_rates():
    report = asset_error_analysis(Y, PREDICTED, GROUPS, NAMES)
    global_metrics = classification_metrics(Y, np.eye(len(NAMES))[PREDICTED], NAMES)
    assert report["schema"] == "servicerobot.diagnosis.asset_errors.v1"
    assert report["class_order"] == NAMES
    assert report["unit"] == "sensor_windows"
    assert report["grouping"] == "dataset_robot_id"
    assert report["used_for_selection"] is False
    assert report["asset_count"] == 4
    overall = report["overall"]
    assert overall["rows"] == 9 and overall["correct_rows"] == 4
    assert overall["normal_rows"] == 3 and overall["normal_false_alarm_rows"] == 1
    assert overall["fault_rows"] == 6 and overall["fault_miss_rows"] == 2
    for key in ("accuracy", "normal_false_alarm_rate", "fault_miss_rate", "confusion_matrix"):
        assert overall[key] == global_metrics[key]
    assert overall["class_support"] == {"fault-a": 3, "정상": 3, "fault-b": 3, "absent-fault": 0}
    assert overall["fault_misses_by_class"] == {
        "fault-a": {"actual_rows": 3, "missed_as_normal_rows": 1, "miss_rate": .333333},
        "fault-b": {"actual_rows": 3, "missed_as_normal_rows": 1, "miss_rate": .333333},
        "absent-fault": {"actual_rows": 0, "missed_as_normal_rows": 0, "miss_rate": None}}
    assert report["assets_with_fault_misses"] == 2
    assert report["assets_with_normal_false_alarms"] == 1
    rows = list(report["per_asset"].values())
    np.testing.assert_array_equal(np.sum([row["confusion_matrix"] for row in rows], axis=0),
                                  overall["confusion_matrix"])
    for key in ("rows", "correct_rows", "normal_rows", "normal_false_alarm_rows", "fault_rows", "fault_miss_rows"):
        assert sum(row[key] for row in rows) == overall[key]
    assert "window_counts_not_independent_failure_episodes" in report["limitations"]
    assert "descriptive_not_causal_or_significance_test" in report["limitations"]
    json.dumps(report, allow_nan=False)


def test_absent_denominators_are_null_not_perfect_or_zero_error_claims():
    report = asset_error_analysis(Y, PREDICTED, GROUPS, NAMES)
    fault_only, normal_only = report["per_asset"]["B"], report["per_asset"]["C"]
    assert fault_only["normal_rows"] == 0
    assert fault_only["normal_false_alarm_rows"] == 0
    assert fault_only["normal_false_alarm_rate"] is None
    assert fault_only["fault_miss_rate"] == .5
    assert normal_only["fault_rows"] == 0
    assert normal_only["fault_miss_rows"] == 0
    assert normal_only["fault_miss_rate"] is None
    assert normal_only["normal_false_alarm_rate"] == 0
    assert normal_only["fault_misses_by_class"]["fault-a"]["miss_rate"] is None


def test_wrong_fault_class_is_not_a_fault_predicted_as_normal():
    counts = asset_error_analysis(Y, PREDICTED, GROUPS, NAMES)["per_asset"]["D"]
    assert counts["rows"] == counts["fault_rows"] == 2
    assert counts["accuracy"] == .5
    assert counts["fault_miss_rows"] == 0
    assert counts["fault_miss_rate"] == 0
    assert counts["confusion_matrix"][2][0] == 1


def test_grouped_view_exposes_concentration_hidden_by_the_same_global_metrics():
    original = asset_error_analysis(Y, PREDICTED, GROUPS, NAMES)
    concentrated = np.where((Y != 1) & (PREDICTED == 1), "miss-robot", "other-robot")
    regrouped = asset_error_analysis(Y, PREDICTED, concentrated, NAMES)
    assert original["overall"] == regrouped["overall"]
    assert original["assets_with_fault_misses"] == 2
    assert regrouped["assets_with_fault_misses"] == 1


def test_order_invariance_and_inputs_remain_unmodified():
    inputs = [Y.copy(), PREDICTED.copy(), GROUPS.copy()]
    before = [value.copy() for value in inputs]
    names = list(NAMES)
    report = asset_error_analysis(*inputs, names)
    assert asset_error_analysis(Y[::-1], PREDICTED[::-1], GROUPS[::-1], names) == report
    assert asset_error_analysis(Y, PREDICTED, GROUPS.astype("S"), names) == report
    for value, original in zip(inputs, before):
        np.testing.assert_array_equal(value, original)
    assert names == NAMES


@pytest.mark.parametrize("order", [[1, 0, 2, 3], [0, 1, 2, 3], [0, 2, 3, 1]])
def test_normal_class_may_occupy_any_index(order):
    inverse = np.argsort(order)
    names = [NAMES[i] for i in order]
    report = asset_error_analysis(inverse[Y], inverse[PREDICTED], GROUPS, names)
    baseline = asset_error_analysis(Y, PREDICTED, GROUPS, NAMES)
    for key in ("rows", "normal_rows", "normal_false_alarm_rows", "normal_false_alarm_rate", "fault_rows",
                "fault_miss_rows", "fault_miss_rate", "fault_misses_by_class", "class_support"):
        assert report["overall"][key] == baseline["overall"][key]
    assert report["class_order"] == names


@pytest.mark.parametrize("labels,predicted,groups", [
    ([0.5, 1], [0, 1], ["a", "b"]), ([0, 1], [0., 1.], ["a", "b"]),
    ([True, False], [0, 1], ["a", "b"]), ([0, 1], [True, False], ["a", "b"]),
    ([-1, 1], [0, 1], ["a", "b"]), ([0, 1], [0, 4], ["a", "b"]),
    ([[0, 1]], [0, 1], ["a", "b"]), ([0, 1], [[0, 1]], ["a", "b"]),
    ([0, 1], [0], ["a", "b"]), ([0, 1], [0, 1], ["a"]), ([], [], []),
    ([0, 1], [0, 1], [1, 2]), ([0, 1], [0, 1], [None, "b"]),
    ([0, 1], [0, 1], ["", "b"]), ([0, 1], [0, 1], [" a", "b"]),
    ([0, 1], [0, 1], ["NaN", "b"]),
])
def test_invalid_or_ambiguous_rows_are_rejected(labels, predicted, groups):
    with pytest.raises(ValueError):
        asset_error_analysis(labels, predicted, groups, NAMES)


@pytest.mark.parametrize("names", [None, [], ["정상"], ["fault", "fault"],
                                    ["정상", "정상"], ["정상", " "], ["정상", None]])
def test_class_order_contract_is_required(names):
    with pytest.raises(ValueError, match="unique class names"):
        asset_error_analysis([0], [0], ["robot"], names)
