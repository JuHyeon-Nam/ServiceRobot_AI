"""Audit Training raw sensor inputs, separating unassigned history from target context."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import zipfile

from audit_diagnosis_continuity import _fingerprints
from audit_diagnosis_source import MAX_JSON_BYTES, _asset_valid, _clock
from build_enhanced_dataset import load_encoding_metadata, parse_record
from diagnosis_record_quality import record_quality


def audit_quality(base_dir, encoding_meta):
    start = time.monotonic()
    metadata, encoding_audit = load_encoding_metadata(encoding_meta)
    folder = base_dir / "Training/02.라벨링데이터"
    before = _fingerprints(folder)
    if not before:
        raise FileNotFoundError("no Training label archives")
    buckets = {name: {"records": 0, "unknown_target_label_records": 0,
                      "dynamic_usable_records": 0, "target_context_usable_records": 0,
                      "all_inputs_usable_records": 0, "dynamic_issues": Counter(), "context_issues": Counter()}
               for name in ("assigned", "unassigned", "missing_label")}
    malformed, examples, non_json = Counter(), [], 0
    seen, clocks, duplicate_keys = defaultdict(set), defaultdict(set), set()
    clock_counts = Counter(dict.fromkeys(("invalid_asset_records", "invalid_timestamp_records",
                                         "duplicate_extra_records", "timestamped_records"), 0))
    for index, info in enumerate(before, 1):
        path = folder / info["archive"]
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.filename.endswith(".json"):
                    non_json += 1
                    continue
                if member.file_size > MAX_JSON_BYTES:
                    raise ValueError(f"JSON exceeds budget: {path.name}::{member.filename}")
                payload = archive.read(member)
                try:
                    document = json.loads(payload.decode("utf-8"))
                    record = parse_record(document)
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError) as error:
                    reason = type(error).__name__
                    malformed[reason] += 1
                    if len(examples) < 5:
                        examples.append({"archive": path.name, "member": member.filename, "reason": reason})
                    continue
                code = record["errorCode"]
                kind = "missing_label" if code is None else "unassigned" if code == "NOT_ASSIGNED" else "assigned"
                result, bucket = record_quality(document, record, metadata), buckets[kind]
                bucket["records"] += 1
                bucket["dynamic_usable_records"] += int(result["dynamic_usable"])
                bucket["target_context_usable_records"] += int(result["target_context_usable"])
                bucket["all_inputs_usable_records"] += int(result["dynamic_usable"] and result["target_context_usable"])
                for group in ("dynamic_issues", "context_issues"):
                    bucket[group].update(f"{name}:{reason}" for name, reason in result[group].items())
                if kind == "missing_label":
                    continue
                if kind == "assigned" and (not isinstance(code, str) or code not in metadata["err_map"]):
                    # Label validity is separate from sensor/context validity.
                    bucket["unknown_target_label_records"] += 1
                asset = record["deviceId"]
                if not _asset_valid(asset):
                    clock_counts["invalid_asset_records"] += 1
                    continue
                try:
                    basis, stamp = _clock(record["createdAt"])
                except (ValueError, TypeError, OverflowError):
                    clock_counts["invalid_timestamp_records"] += 1
                    continue
                key = (basis, stamp)
                if key in seen[asset]:
                    clock_counts["duplicate_extra_records"] += 1
                    duplicate_keys.add((asset, *key))
                seen[asset].add(key)
                clocks[asset].add(basis)
                clock_counts["timestamped_records"] += 1
        print(f"[quality] {index}/{len(before)} ZIP", file=sys.stderr, flush=True)
    if before != _fingerprints(folder):
        raise ValueError("Training archives changed during quality audit")
    clock = {**clock_counts, "assets": len(seen), "duplicate_timestamp_groups": len(duplicate_keys),
             "mixed_clock_assets": sorted(asset for asset, bases in clocks.items() if len(bases) > 1)}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    files = ("audit_diagnosis_quality.py", "diagnosis_record_quality.py", "audit_diagnosis_source.py",
             "audit_diagnosis_continuity.py", "build_enhanced_dataset.py")
    return {"schema": "servicerobot.diagnosis.input_quality.v1", "scope": "training_raw_field_quality_only",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(), "encoding_audit": encoding_audit,
            "buckets": buckets, "malformed_records": dict(malformed), "malformed_examples": examples,
            "json_records": sum(b["records"] for b in buckets.values()) + sum(malformed.values()),
            "non_json_members": non_json, "clock": clock,
            "official_validation_read": False, "sensor_window_arrays_created": False,
            "model_trained": False, "model_performance_evaluated": False, "training_readiness": "not_evaluated",
            "lineage": {"training_archives": before, "code_sha256": {
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in files}},
            "resources": {"elapsed_seconds": round(time.monotonic() - start, 3),
                          "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024},
            "limitations": ["representability_and_presence_not_physical_accuracy_or_sensor_calibration",
                            "numeric_strings_and_booleans_follow_existing_float_parser",
                            "target_label_validity_is_separate_from_sensor_context_counts",
                            "clock_checks_use_assigned_and_unassigned_records; missing_labels_excluded",
                            "no_window_continuity_or_robot_holdout_readiness_decision",
                            "timestamp_keys_retained_in_memory; no_constant_memory_guarantee",
                            "frozen_metadata_may_include_historical_validation_information",
                            "source_authenticity_and_atomic_snapshot_not_verified"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--encoding-meta", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        parser.exit(2, "Quality audit: report must be a new file\n")
    try:
        report = audit_quality(args.base_dir, args.encoding_meta)
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    except (OSError, ValueError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Quality audit failed: {error}\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
