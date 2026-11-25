"""Synthetic split-boundary checks; no consent, model fitting, or real-data evaluation."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnosis_temporal import history_span_bound, purged_temporal_holdout, temporal_inputs


def times(seconds):
    return np.asarray([(pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(seconds=s)).isoformat()
                       for s in seconds])


def metadata():
    return {"window_contract": {"history": "30 observations before target", "context": "at target time"},
            "sequence_audit": {"max_gap_seconds": 10, "duplicate_policy": "reject",
                               "timestamp_policy": "consistent_asset_clock; aware_times_normalized_to_utc"}}


def test_global_boundary_purge_equality_unseen_robots_and_input_order():
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "new"])
    stamps = times([0, 10, 20, 21, 0, 10, 31, 40])
    before_groups, before_stamps = groups.copy(), stamps.copy()
    fit, selection, report = purged_temporal_holdout(groups, stamps, cutoff=times([10])[0], history_span_ns=20 * 10**9)
    assert fit.tolist() == [0, 4] and selection.tolist() == [3, 6]
    assert report["purged_rows"] == 3 and report["unseen_future_rows"] == 1
    assert report["future_candidate_rows"] == 6
    assert report["rows"] == len(fit) + len(selection) + report["purged_rows"] + report["unseen_future_rows"]
    assert report["declared_interval_bounds_disjoint"] and report["interval_guard_checked_rows"] == 2
    assert not report["labels_used_for_split"] and not report["split_search_performed"]
    assert np.array_equal(groups, before_groups) and np.array_equal(stamps, before_stamps)


def test_no_global_future_to_past_reversal_and_no_overlapping_closed_intervals():
    rng = np.random.default_rng(42)
    groups = np.repeat(["a", "b", "c"], 100)
    seconds = np.tile(np.arange(100), 3)
    order = rng.permutation(len(groups))
    groups, stamps = groups[order], times(seconds[order])
    fit, selected, report = purged_temporal_holdout(groups, stamps, cutoff=times([60])[0], history_span_ns=10 * 10**9)
    values = pd.to_datetime(stamps, utc=True).as_unit("ns").asi8
    assert values[fit].max() < values[selected].min()
    for selection in selected:
        same_asset_fit = fit[groups[fit] == groups[selection]]
        assert values[selection] - 10 * 10**9 > values[same_asset_fit].max()
    assert report["purged_rows"] == 30


@pytest.mark.parametrize("cut", [-10, 100])
def test_empty_sides_are_reported_without_fabricated_guard_evidence(cut):
    fit, selected, report = purged_temporal_holdout(["a"] * 3, times([0, 10, 20]),
        cutoff=times([cut])[0], history_span_ns=10**9)
    assert not report["both_sides_nonempty"] and not len(selected)
    assert report["declared_interval_bounds_disjoint"] is None and report["interval_guard_checked_rows"] == 0


def test_timezone_normalization_and_nanosecond_boundary_are_exact():
    values = ["2026-01-01T09:00:00+09:00", "2026-01-01T00:00:00.000000001Z",
              "2026-01-01T00:00:00.000000002Z"]
    fit, selection, report = purged_temporal_holdout(["a"] * 3, values,
        cutoff=values[1], history_span_ns=1)
    assert fit.tolist() == [0] and selection.tolist() == [2] and report["purged_rows"] == 1


@pytest.mark.parametrize("bad", [None, "", "bad", "2026-01-01T00:00:00", " 2026-01-01T00:00:00Z"])
def test_ambiguous_or_invalid_times_are_rejected(bad):
    with pytest.raises(ValueError):
        temporal_inputs(["a", "a"], ["2026-01-01T00:00:00Z", bad])


def test_duplicate_instants_with_different_offsets_are_rejected_per_robot():
    with pytest.raises(ValueError, match="duplicate normalized"):
        temporal_inputs(["a", "a"], ["2026-01-01T09:00:00+09:00", "2026-01-01T00:00:00Z"])
    assert len(temporal_inputs(["a", "b"], times([0, 0]))[1]) == 2


@pytest.mark.parametrize("groups,stamps", [([], []), ([1], ["2026-01-01T00:00:00Z"]),
    (["nan"], ["2026-01-01T00:00:00Z"]), ([" a"], ["2026-01-01T00:00:00Z"]),
    (["a", "b"], ["2026-01-01T00:00:00Z"]), ([["a"]], [["2026-01-01T00:00:00Z"]])])
def test_invalid_robot_and_alignment_inputs_are_rejected(groups, stamps):
    with pytest.raises(ValueError):
        temporal_inputs(groups, stamps)


@pytest.mark.parametrize("span", [None, 0, -1, True, 1.5, 2**63])
def test_invalid_span_rejected(span):
    with pytest.raises(ValueError):
        purged_temporal_holdout(["a"], times([0]), cutoff=times([1])[0], history_span_ns=span)


def test_history_lower_bound_overflow_is_rejected():
    with pytest.raises(ValueError, match="underflow"):
        purged_temporal_holdout(["a"], ["1678-01-01T00:00:00Z"],
            cutoff=times([1])[0], history_span_ns=np.iinfo(np.int64).max)


def test_history_bound_uses_all_thirty_gaps_including_target_and_preserves_metadata():
    meta = metadata()
    before = deepcopy(meta)
    assert history_span_bound(meta)["history_span_upper_bound_ns"] == 300 * 10**9
    meta["sequence_audit"]["max_gap_seconds"] = 1e-9
    assert history_span_bound(meta)["history_span_upper_bound_ns"] == 30
    meta["sequence_audit"]["max_gap_seconds"] = 10
    assert meta == before


@pytest.mark.parametrize("meta", [None, [], {"window_contract": None},
    {"window_contract": metadata()["window_contract"], "sequence_audit": None}])
def test_invalid_metadata_shapes_fail_closed(meta):
    with pytest.raises(ValueError):
        history_span_bound(meta)


@pytest.mark.parametrize("issue", ["window", "context", "clock", "duplicate", "no_gap", "bad_gap", "overflow"])
def test_missing_or_incompatible_history_contract_is_not_assumed_safe(issue):
    meta = metadata()
    if issue == "window":
        meta["window_contract"]["history"] = "30 observations including target"
    elif issue == "context":
        meta["window_contract"]["context"] = "unknown"
    elif issue == "clock":
        meta["sequence_audit"]["timestamp_policy"] = "unknown"
    elif issue == "duplicate":
        meta["sequence_audit"]["duplicate_policy"] = "allow"
    else:
        meta["sequence_audit"]["max_gap_seconds"] = {"no_gap": None, "bad_gap": 0, "overflow": 1e9}[issue]
    with pytest.raises(ValueError):
        history_span_bound(meta)
