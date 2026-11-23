"""Necessary allocation bounds on synthetic windows, not a split solver or model evaluation."""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnosis_evaluation import class_holdout_bounds

NAMES = ["fault", "정상"]


def arrays(counts):
    labels, groups = [], []
    for robot, count in enumerate(counts):
        for code in range(2):
            labels.extend([code] * count)
            groups.extend([f"robot-{robot}"] * count)
    return np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=str)


def test_two_fault_robots_cannot_supply_one_fit_and_two_selection_robots():
    report = class_holdout_bounds(*arrays([1000, 1000]), NAMES)
    row = report["per_class"]["fault"]
    assert row["rows"] == 2000 and row["assets"] == 2
    assert row["asset_deficit"] == 1 and row["required_total_assets"] == 3
    assert not report["necessary_conditions_passed"]
    assert report["joint_split_feasible"] is None and not report["split_search_performed"]


def test_exact_boundary_counts_pass_without_claiming_joint_feasibility():
    y, groups = arrays([1, 10, 10])
    before_y, before_groups = y.copy(), groups.copy()
    report = class_holdout_bounds(y, groups, NAMES)
    row = report["per_class"]["fault"]
    assert report["necessary_conditions_passed"] and report["joint_split_feasible"] is None
    assert row["selection_rows_upper_bound"] == 20 and row["row_deficit"] == row["asset_deficit"] == 0
    assert row["rows_by_asset"] == {"robot-0": 1, "robot-1": 10, "robot-2": 10}
    assert np.array_equal(y, before_y) and np.array_equal(groups, before_groups)


@pytest.mark.parametrize("side", ["fit", "selection"])
def test_reserving_robot_assets_tightens_the_opposite_rows_upper_bound(side):
    policy = {"min_selection_rows": 101, "min_selection_assets": 1, "min_fit_rows": 1, "min_fit_assets": 2}
    if side == "fit":
        policy = {"min_selection_rows": 1, "min_selection_assets": 2, "min_fit_rows": 101, "min_fit_assets": 1}
    report = class_holdout_bounds(*arrays([100, 1, 1]), NAMES, **policy)
    row = report["per_class"]["fault"]
    assert row["row_deficit"] == row["asset_deficit"] == 0
    assert row[f"{side}_rows_upper_bound"] == 100
    assert any(i["code"] == f"{side}_rows_exceed_asset_preserving_upper_bound" for i in row["issues"])


def test_insufficient_total_rows_and_missing_class_do_not_fabricate_bounds():
    report = class_holdout_bounds(*arrays([2, 3, 4]), NAMES, min_selection_rows=10)
    assert report["per_class"]["fault"]["row_deficit"] == 2
    empty = class_holdout_bounds(np.array([], dtype=np.int64), np.array([], dtype=str), NAMES)
    for row in empty["per_class"].values():
        assert row["selection_rows_upper_bound"] is row["fit_rows_upper_bound"] is None
        assert row["asset_deficit"] == 3 and not row["necessary_conditions_passed"]


def test_passing_necessary_conditions_can_still_have_no_possible_partition():
    y, groups = arrays([4, 4, 4])
    report = class_holdout_bounds(y, groups, NAMES, min_selection_rows=5, min_selection_assets=1, min_fit_rows=5)
    assert report["necessary_conditions_passed"] and report["joint_split_feasible"] is None
    totals = [sum(4 * bit for bit in bits) for bits in product([0, 1], repeat=3)]
    assert not any(selection >= 5 and 12 - selection >= 5 for selection in totals)


@pytest.mark.parametrize("counts", [[1, 2, 5], [2, 2, 2, 2], [1, 1, 8, 8], [4, 4, 4]])
def test_bounds_never_reject_a_small_brute_force_feasible_partition(counts):
    y, groups = arrays(counts)
    for selection_rows, fit_rows, selection_assets, fit_assets in product((1, 3, 5), (1, 3, 5), (1, 2), (1, 2)):
        report = class_holdout_bounds(y, groups, NAMES, min_selection_rows=selection_rows,
            min_selection_assets=selection_assets, min_fit_rows=fit_rows, min_fit_assets=fit_assets)
        feasible = any(sum(bits) >= selection_assets and len(bits) - sum(bits) >= fit_assets
                       and sum(count * bit for count, bit in zip(counts, bits)) >= selection_rows
                       and sum(count * (1 - bit) for count, bit in zip(counts, bits)) >= fit_rows
                       for bits in product((0, 1), repeat=len(counts)))
        if feasible:
            assert report["necessary_conditions_passed"]


@pytest.mark.parametrize("policy", [{"min_fit_rows": 0}, {"min_fit_assets": True},
    {"min_fit_rows": 1.5}, {"min_selection_assets": -1}, {"min_selection_rows": "20"}])
def test_invalid_policies_rejected(policy):
    with pytest.raises(ValueError):
        class_holdout_bounds(*arrays([1, 10, 10]), NAMES, **policy)


@pytest.mark.parametrize("issue", ["label", "group", "alignment", "names"])
def test_existing_label_robot_and_class_guards_are_reused(issue):
    y, groups = arrays([1, 10, 10])
    names = NAMES
    if issue == "label":
        y[0] = 2
    elif issue == "group":
        groups[0] = "nan"
    elif issue == "alignment":
        groups = groups[:-1]
    else:
        names = ["fault", "fault"]
    with pytest.raises(ValueError):
        class_holdout_bounds(y, groups, names)
