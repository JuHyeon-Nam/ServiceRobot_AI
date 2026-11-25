"""Label-blind temporal split preparation, separate from the existing robot holdout."""
from numbers import Integral

import numpy as np
import pandas as pd

from build_enhanced_dataset import WIN, _gap_threshold


def history_span_bound(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("temporal history metadata must be an object")
    contract, sequence = metadata.get("window_contract", {}), metadata.get("sequence_audit", {})
    if not isinstance(contract, dict) or not isinstance(sequence, dict):
        raise ValueError("temporal window and sequence contracts must be objects")
    if contract != {"history": f"{WIN} observations before target", "context": "at target time"}:
        raise ValueError("temporal preparation requires the declared 30-history/target-context contract")
    if (sequence.get("duplicate_policy") != "reject" or
            sequence.get("timestamp_policy") != "consistent_asset_clock; aware_times_normalized_to_utc"):
        raise ValueError("temporal preparation requires the builder clock and duplicate policy")
    threshold = _gap_threshold(sequence.get("max_gap_seconds"))
    if threshold is None:
        raise ValueError("an explicit positive adjacent-gap bound is required")
    span = WIN * int(threshold.value)
    if span > np.iinfo(np.int64).max:
        raise ValueError("history span bound exceeds int64 nanoseconds")
    return {"history_observations": WIN, "max_adjacent_gap_ns": int(threshold.value),
            "history_span_upper_bound_ns": span,
            "basis": "declared_metadata_including_target_gap_not_independently_verified_history_times"}


def _utc_ns(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("temporal times must be trimmed ISO8601 strings")
    try:
        stamp = pd.to_datetime(value, format="ISO8601", errors="raise")
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("a global cutoff requires timezone-aware timestamps")
        return int(stamp.tz_convert("UTC").as_unit("ns").value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"invalid timezone-aware ISO8601 time: {value!r}") from error


def temporal_inputs(groups, target_times):
    groups, target_times = np.asarray(groups), np.asarray(target_times)
    if (groups.ndim != 1 or not len(groups) or groups.shape != target_times.shape
            or groups.dtype.kind not in "US" or target_times.dtype.kind not in "US"):
        raise ValueError("temporal preparation needs aligned nonempty string robot IDs and target times")
    groups, target_times = groups.astype(str), target_times.astype(str)
    if any(not g.strip() or g != g.strip() or g.lower() in {"nan", "none", "null", "nat"} for g in groups):
        raise ValueError("temporal preparation requires canonical robot IDs")
    stamps = np.asarray([_utc_ns(value) for value in target_times], dtype=np.int64)
    for asset in np.unique(groups):
        ordered = np.sort(stamps[groups == asset])
        if len(ordered) > 1 and (ordered[1:] == ordered[:-1]).any():
            raise ValueError(f"duplicate normalized robot/target time: {asset}")
    return groups, stamps


def purged_temporal_holdout(groups, target_times, *, cutoff, history_span_ns):
    """Use one global cutoff, retain only seen robots, and purge closed-interval overlap."""
    if (not isinstance(history_span_ns, Integral) or isinstance(history_span_ns, bool)
            or not 1 <= history_span_ns <= np.iinfo(np.int64).max):
        raise ValueError("history_span_ns must be a positive int64 duration")
    cutoff_ns = _utc_ns(cutoff)
    groups, stamps = temporal_inputs(groups, target_times)
    span = int(history_span_ns)
    if int(stamps.min()) < np.iinfo(np.int64).min + span:
        raise ValueError("history lower bound would underflow int64 nanoseconds")
    fit_mask, future = stamps < cutoff_ns, stamps >= cutoff_ns
    fit_assets = np.unique(groups[fit_mask])
    known = np.isin(groups, fit_assets)
    lower = stamps - span
    purged = np.zeros(len(stamps), dtype=bool)
    per_asset = {}
    for asset in np.unique(groups):
        mask = groups == asset
        fit_times = stamps[mask & fit_mask]
        last_fit = int(fit_times.max()) if len(fit_times) else None
        if last_fit is not None:
            purged |= mask & future & (lower <= last_fit)
        selected = mask & future & known & ~purged
        per_asset[str(asset)] = {"fit_rows": int((mask & fit_mask).sum()),
            "selection_rows": int(selected.sum()), "purged_rows": int((mask & purged).sum()),
            "unseen_future_rows": int((mask & future & ~known).sum()),
            "latest_fit_target_ns": last_fit,
            "earliest_selection_history_lower_bound_ns": int(lower[selected].min()) if selected.any() else None}
    selection_mask = future & known & ~purged
    fit, selection = np.flatnonzero(fit_mask), np.flatnonzero(selection_mask)
    for row in per_asset.values():
        if row["selection_rows"]:
            if row["earliest_selection_history_lower_bound_ns"] <= row["latest_fit_target_ns"]:
                raise ValueError("cross-split history interval overlap")
    report = {"schema": "servicerobot.diagnosis.temporal_plan.v1",
        "cutoff_utc": pd.Timestamp(cutoff_ns, unit="ns", tz="UTC").isoformat(),
        "history_span_upper_bound_ns": span, "rows": len(stamps), "fit_rows": len(fit),
        "future_candidate_rows": int(future.sum()), "selection_rows": len(selection),
        "purged_rows": int(purged.sum()), "unseen_future_rows": int((future & ~known).sum()),
        "both_sides_nonempty": bool(len(fit) and len(selection)),
        "declared_interval_bounds_disjoint": True if len(selection) else None,
        "interval_guard_checked_rows": len(selection), "per_asset": per_asset,
        "labels_used_for_split": False, "split_search_performed": False,
        "scope": "existing_robots_later_than_one_global_cutoff_proposal",
        "limitations": ["not_new_robot_generalization_or_official_validation",
                        "declared_history_bound_not_independently_verified_raw_history_intervals",
                        "no_class_coverage_or_support_guarantee; no_model_or_training_readiness_decision",
                        "within_split_windows_can_overlap_and_are_not_independent_fault_events",
                        "same_robot_future_only; sensor_clock_authenticity_not_verified"]}
    return fit, selection, report
