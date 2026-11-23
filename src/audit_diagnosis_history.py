"""Compare label-filtered and unassigned-inclusive timestamp histories on Training only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from audit_diagnosis_continuity import _fingerprints, validate_probes
from audit_diagnosis_source import MAX_JSON_BYTES, _asset_valid
from build_enhanced_dataset import WIN, load_encoding_metadata, parse_record, prepare_sequence


def read_histories(folder, metadata):
    archives = sorted(folder.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"no Training label archives in {folder}")
    by_asset = defaultdict(list)
    counts = {"json_records": 0, "missing_label_excluded": 0, "unassigned_history_records": 0,
              "assigned_target_records": 0, "non_json_members": 0}
    for index, path in enumerate(archives, 1):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.filename.endswith(".json"):
                    counts["non_json_members"] += 1
                    continue
                if member.file_size > MAX_JSON_BYTES:
                    raise ValueError(f"JSON member exceeds budget: {path.name}::{member.filename}")
                payload = archive.read(member)
                try:
                    record = parse_record(json.loads(payload.decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError) as error:
                    raise ValueError(f"malformed record: {path.name}::{member.filename}") from error
                counts["json_records"] += 1
                code = record["errorCode"]
                if code is None:
                    counts["missing_label_excluded"] += 1
                    continue
                if not _asset_valid(record["deviceId"]):
                    raise ValueError("history audit requires canonical robot IDs")
                unassigned = code == "NOT_ASSIGNED"
                if not unassigned:
                    for field, mapping in (("errorCode", "err_map"), ("deviceType", "devtype_map"),
                                           ("mainState", "mainstate_map")):
                        value = record[field]
                        if not isinstance(value, str) or value not in metadata[mapping]:
                            raise ValueError(f"unknown {field} {value!r}; frozen encoding")
                counts["unassigned_history_records" if unassigned else "assigned_target_records"] += 1
                by_asset[record["deviceId"]].append({"createdAt": record["createdAt"], "errorCode": code})
        print(f"[history] {index}/{len(archives)} ZIP; {counts['json_records']} JSON", file=sys.stderr, flush=True)
    return by_asset, counts


def _window_gaps(stamps):
    deltas = np.diff(stamps)
    if (deltas <= 0).any():
        raise ValueError("intervals must be positive int64 nanoseconds")
    spans = stamps[WIN:] - stamps[:-WIN]
    if (spans <= 0).any():
        raise ValueError("window spans must be positive int64 nanoseconds")
    result = np.full(len(stamps), np.iinfo(np.int64).max, dtype=np.int64)
    if len(stamps) > WIN:
        result[WIN:] = np.lib.stride_tricks.sliding_window_view(deltas, WIN).max(axis=1)
    return result


def compare_histories(by_asset, metadata, gap_seconds):
    probes = validate_probes(gap_seconds)
    names = [name for name, _ in sorted(metadata["err_map"].items(), key=lambda item: item[1])]
    support = {name: dict.fromkeys(("target_records", "baseline_candidate_windows",
                                    "history_only_candidate_windows", "unassigned_in_history_targets"), 0)
               for name in names}
    results = [{"max_gap_seconds": seconds, "per_class": {name: dict.fromkeys(
        ("both_retained", "baseline_only_retained", "history_only_retained", "neither_retained"), 0)
        for name in names}} for seconds, _ in probes]
    robots = [[defaultdict(set), defaultdict(set)] for _ in probes]
    for asset, records in by_asset.items():
        ordered, _, clock = prepare_sequence(records, asset)
        stamps = pd.to_datetime([r["createdAt"].strip() for r in ordered], format="ISO8601",
                                utc=clock["timestamp_basis"] == "utc").as_unit("ns").asi8
        codes = [r["errorCode"] for r in ordered]
        if any(code != "NOT_ASSIGNED" and code not in metadata["err_map"] for code in codes):
            raise ValueError("unknown assigned class in history")
        assigned = np.asarray([code != "NOT_ASSIGNED" for code in codes])
        positions = np.flatnonzero(assigned)
        labels = np.asarray([metadata["err_map"][codes[i]] for i in positions], dtype=np.int64)
        baseline_gaps, inclusive_gaps = _window_gaps(stamps[positions]), _window_gaps(stamps)[positions]
        baseline_ready, inclusive_ready = np.arange(len(positions)) >= WIN, positions >= WIN
        prefix = np.concatenate(([0], np.cumsum(~assigned)))
        touched = inclusive_ready & (prefix[positions] - prefix[np.maximum(0, positions - WIN)] > 0)
        for name in names:
            mask = labels == metadata["err_map"][name]
            for field, values in (("target_records", mask), ("baseline_candidate_windows", mask & baseline_ready),
                                  ("history_only_candidate_windows", mask & inclusive_ready),
                                  ("unassigned_in_history_targets", mask & touched)):
                support[name][field] += int(values.sum())
        for index, (_, threshold) in enumerate(probes):
            baseline = baseline_ready & (baseline_gaps <= threshold)
            inclusive = inclusive_ready & (inclusive_gaps <= threshold)
            for name in names:
                mask = labels == metadata["err_map"][name]
                for field, values in (("both_retained", baseline & inclusive),
                                      ("baseline_only_retained", baseline & ~inclusive),
                                      ("history_only_retained", ~baseline & inclusive),
                                      ("neither_retained", ~baseline & ~inclusive)):
                    results[index]["per_class"][name][field] += int((mask & values).sum())
                for mode, values in enumerate((baseline, inclusive)):
                    if (mask & values).any():
                        robots[index][mode][name].add(asset)
    for index, result in enumerate(results):
        for name, row in result["per_class"].items():
            row.update({"baseline_retained": row["both_retained"] + row["baseline_only_retained"],
                        "inclusive_retained": row["both_retained"] + row["history_only_retained"],
                        "baseline_assets": len(robots[index][0][name]),
                        "inclusive_assets": len(robots[index][1][name])})
        result["baseline_retained"] = sum(row["baseline_retained"] for row in result["per_class"].values())
        result["inclusive_retained"] = sum(row["inclusive_retained"] for row in result["per_class"].values())
    return {"per_class": support, "probes": results,
            "paired_comparison_denominator": "all_assigned_target_records_including_short_histories"}


def audit_history(base_dir, encoding_meta, gap_seconds=(5, 10, 60)):
    validate_probes(gap_seconds)
    start = time.monotonic()
    metadata, encoding_audit = load_encoding_metadata(encoding_meta)
    folder = base_dir / "Training/02.라벨링데이터"
    before = _fingerprints(folder)
    records, ingestion = read_histories(folder, metadata)
    result = compare_histories(records, metadata, gap_seconds)
    if before != _fingerprints(folder):
        raise ValueError("Training archives changed during history audit")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    files = ("audit_diagnosis_history.py", "audit_diagnosis_continuity.py", "audit_diagnosis_source.py",
             "build_enhanced_dataset.py")
    return {"schema": "servicerobot.diagnosis.history_audit.v1", "scope": "training_timestamp_counterfactual_only",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(), "ingestion": ingestion, **result,
            "encoding_audit": encoding_audit, "history_observations": WIN, "target_in_gap_check": True,
            "unassigned_used_as_target": False, "missing_label_history_included": False,
            "official_validation_read": False, "sensor_window_arrays_created": False,
            "model_trained": False, "model_performance_evaluated": False, "policy_selected": False,
            "lineage": {"training_archives": before, "code_sha256": {
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in files}},
            "resources": {"elapsed_seconds": round(time.monotonic() - start, 3),
                          "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024},
            "limitations": ["timestamp_only_counterfactual_not_a_prepared_or_approved_training_dataset",
                            "unassigned_sensor_values_and_context_quality_not_validated",
                            "unassigned_labels_never_imputed_as_normal",
                            "all_compact_timestamps_retained_in_memory; not_constant_memory",
                            "windows_not_independent_fault_events; no_model_or_causal_claim",
                            "frozen_metadata_may_include_historical_validation_information",
                            "source_authenticity_and_atomic_snapshot_not_verified"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--encoding-meta", required=True, type=Path)
    parser.add_argument("--gap-seconds", nargs="+", type=float, default=[5, 10, 60])
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        parser.exit(2, "History audit: report must be a new file\n")
    try:
        report = audit_history(args.base_dir, args.encoding_meta, args.gap_seconds)
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    except (OSError, ValueError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as error:
        parser.exit(2, f"History audit failed: {error}\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
