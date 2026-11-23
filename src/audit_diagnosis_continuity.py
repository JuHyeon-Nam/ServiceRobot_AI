"""Inspect Training-only observation gaps and window attrition without training or feature arrays."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import json
import resource
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from build_enhanced_dataset import (WIN, _gap_threshold, load_encoding_metadata, prepare_sequence, read_split)


def validate_probes(gap_seconds):
    if not gap_seconds or any(seconds is None for seconds in gap_seconds):
        raise ValueError("at least one explicit positive gap probe is required")
    probes = [(float(seconds), int(_gap_threshold(seconds).value)) for seconds in gap_seconds]
    if len({ns for _, ns in probes}) != len(probes):
        raise ValueError("gap probes must have distinct nanosecond thresholds")
    return probes


def _quantiles(values):
    if not len(values):
        return {name: None for name in ("min", "p50", "p95", "p99", "max")}
    return dict(zip(("min", "p50", "p95", "p99", "max"),
                    map(int, np.quantile(values, [0, .5, .95, .99, 1], method="nearest"))))


def profile_sequences(by_asset, metadata, gap_seconds):
    probes = validate_probes(gap_seconds)
    names = [name for name, _ in sorted(metadata["err_map"].items(), key=lambda item: item[1])]
    baseline_rows, baseline_assets = Counter(), defaultdict(set)
    kept_rows = [Counter() for _ in probes]
    kept_assets = [defaultdict(set) for _ in probes]
    gap_edges = [0 for _ in probes]
    intervals, per_asset, max_span = [], {}, None
    for asset, records in by_asset.items():
        for record in records:
            for field, mapping in (("deviceType", "devtype_map"), ("mainState", "mainstate_map"),
                                   ("errorCode", "err_map")):
                value = record[field]
                if not isinstance(value, str) or value not in metadata[mapping]:
                    raise ValueError(f"unknown {field} {value!r} for {asset}; frozen encoding")
        ordered, _, clock = prepare_sequence(records, asset)
        stamps = pd.to_datetime([row["createdAt"].strip() for row in ordered], format="ISO8601",
                                utc=clock["timestamp_basis"] == "utc").as_unit("ns").asi8
        deltas = np.diff(stamps)
        if (deltas <= 0).any():
            raise ValueError("intervals must be positive and representable as int64 nanoseconds")
        intervals.append(deltas)
        window_gaps = (np.lib.stride_tricks.sliding_window_view(deltas, WIN).max(axis=1)
                       if len(ordered) > WIN else np.empty(0, dtype=np.int64))
        spans = stamps[WIN:] - stamps[:-WIN] if len(ordered) > WIN else np.empty(0, dtype=np.int64)
        if (spans <= 0).any():
            raise ValueError("window spans must be representable as int64 nanoseconds")
        if len(spans):
            max_span = max(max_span or 0, int(spans.max()))
        labels = np.asarray([metadata["err_map"][row["errorCode"]] for row in ordered[WIN:]], dtype=np.int64)
        for name in names:
            count = int(np.count_nonzero(labels == metadata["err_map"][name]))
            baseline_rows[name] += count
            if count:
                baseline_assets[name].add(asset)
        for index, (_, threshold) in enumerate(probes):
            mask = window_gaps <= threshold
            gap_edges[index] += int(np.count_nonzero(deltas > threshold))
            for name in names:
                count = int(np.count_nonzero(labels[mask] == metadata["err_map"][name]))
                kept_rows[index][name] += count
                if count:
                    kept_assets[index][name].add(asset)
        per_asset[asset] = {"records": len(ordered), "candidate_windows": len(labels),
                            "timestamp_basis": clock["timestamp_basis"], "adjacent_intervals": len(deltas),
                            "interval_quantiles_ns": _quantiles(deltas)}
    all_intervals = np.concatenate(intervals) if intervals else np.empty(0, dtype=np.int64)
    baseline = {name: {"windows": baseline_rows[name], "assets": len(baseline_assets[name])} for name in names}
    total = sum(baseline_rows.values())
    results = []
    for index, (seconds, threshold) in enumerate(probes):
        kept = sum(kept_rows[index].values())
        results.append({"max_gap_seconds": seconds, "max_gap_ns": threshold, "gap_edges": gap_edges[index],
                        "retained_windows": kept, "discarded_windows": total - kept,
                        "retention_fraction": kept / total if total else None,
                        "per_class": {name: {"retained_windows": kept_rows[index][name],
                                              "discarded_windows": baseline_rows[name] - kept_rows[index][name],
                                              "retained_assets": len(kept_assets[index][name])} for name in names}})
    return {"candidate_windows": total, "baseline_support": baseline, "probes": results,
            "adjacent_intervals": len(all_intervals), "interval_quantile_method": "nearest",
            "interval_quantiles_ns": _quantiles(all_intervals), "max_window_span_ns": max_span,
            "per_asset": per_asset}


def _fingerprints(folder):
    result = []
    for path in sorted(folder.glob("*.zip")):
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        result.append({"archive": path.name, "bytes": path.stat().st_size, "sha256": sha})
    return result


def audit_continuity(base_dir, encoding_meta, gap_seconds=(5, 10, 60)):
    validate_probes(gap_seconds)
    start = time.monotonic()
    metadata, encoding_audit = load_encoding_metadata(encoding_meta)
    folder = base_dir / "Training/02.라벨링데이터"
    before = _fingerprints(folder)
    ingestion = {}
    records, _, _ = read_split(folder, "train", audit=ingestion, reject_malformed_records=True)
    profile = profile_sequences(records, metadata, gap_seconds)
    if before != _fingerprints(folder):
        raise ValueError("Training archives changed during continuity audit")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    code = {}
    for name in ("audit_diagnosis_continuity.py", "build_enhanced_dataset.py"):
        with (Path(__file__).parent / name).open("rb") as stream:
            code[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"schema": "servicerobot.diagnosis.continuity.v1", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "training_source_continuity_only", "encoding_audit": encoding_audit,
            "window_contract": {"history_observations": WIN, "target_in_gap_check": True},
            "ingestion": ingestion, **profile, "policy_selected": False,
            "official_validation_read": False, "sensor_window_arrays_created": False,
            "model_trained": False, "model_performance_evaluated": False,
            "lineage": {"training_archives": before, "code_sha256": code},
            "resources": {"elapsed_seconds": round(time.monotonic() - start, 3),
                          "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024},
            "limitations": ["label_filtered_records_match_existing_builder; unassigned_history_excluded",
                            "probe_comparison_does_not_select_a_policy_or_prove_training_readiness",
                            "window_counts_not_independent_fault_events",
                            "timestamps_and_differences_must_fit_int64_nanoseconds",
                            "source_authenticity_and_atomic_snapshot_not_verified",
                            "all_parsed_training_records_retained; no_sensor_window_arrays_or_features",
                            "frozen_metadata_may_include_historical_validation_information"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--encoding-meta", required=True, type=Path)
    parser.add_argument("--gap-seconds", nargs="+", type=float, default=[5, 10, 60])
    parser.add_argument("--report", required=True, type=Path, help="new JSON report; never overwrite")
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        parser.exit(2, "Continuity audit: report must be a new file\n")
    try:
        report = audit_continuity(args.base_dir, args.encoding_meta, args.gap_seconds)
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    except (OSError, ValueError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Continuity audit failed: {error}\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
