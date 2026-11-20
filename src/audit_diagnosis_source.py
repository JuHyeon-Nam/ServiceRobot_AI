"""Audit recorded diagnosis ZIPs without retaining sensor histories or making windows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sqlite3
import sys
import tempfile
import time
import zipfile

import pandas as pd

from build_enhanced_dataset import (CROWD, INGEST_SKIP_REASONS,
                                    load_encoding_metadata, parse_record)

SPLITS = {"train": "Training", "validation": "Validation"}
MAX_JSON_BYTES = 8 * 1024 * 1024
NUMERIC_PATHS = {
    "batteryLevel": ("deviceData", "batteryLevel"),
    **{key: ("deviceData", "standardLocationData", key) for key in ("speed", "x", "y", "degree")},
    **{key: ("deviceData", key) for key in ("collision", "obstacle", "isOffline", "nowCharging", "emergencyStop")},
    **{key: ("deviceData", "totalOperationData", key) for key in ("batteryUse", "batteryCycleCount", "distance")},
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_valid(value):
    return (isinstance(value, str) and bool(value.strip()) and value == value.strip()
            and value.lower() not in {"none", "null", "nan", "nat"})


def _clock(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing timestamp")
    stamp = pd.to_datetime(value.strip(), format="ISO8601", errors="raise")
    if pd.isna(stamp):
        raise ValueError("missing timestamp")
    aware = stamp.tzinfo is not None
    if aware:
        stamp = stamp.tz_convert("UTC")
    return int(aware), stamp.value


def _numeric_issues(document):
    issues = {}
    for key, path in NUMERIC_PATHS.items():
        value = document
        for part in path:
            value = value.get(part) if isinstance(value, dict) else None
        try:
            number = float(value)
        except (ValueError, TypeError):
            issues[key] = "missing_or_invalid_defaults_to_zero"
            continue
        if not math.isfinite(number) or abs(number) > 3.4028234663852886e38:
            issues[key] = "nonfinite_or_outside_float32"
    return issues


def _scan_split(base, split, database, encodings):
    archives = sorted((base / SPLITS[split] / "02.라벨링데이터").glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"no label JSON archives for {split}")
    counts = Counter(dict.fromkeys(("members", "json_members", "non_json_members", "label_kept_records",
                                   "timestamped_records", "invalid_asset_records", "invalid_timestamp_records",
                                   "out_of_order_arrival_steps", "crowd_defaults_to_middle"), 0))
    skipped = Counter(dict.fromkeys(INGEST_SKIP_REASONS, 0))
    labels, assets, classes_by_asset = Counter(), set(), defaultdict(set)
    invalid_categories, unknown_categories = Counter(), defaultdict(Counter)
    numeric, clocks, last = Counter(), defaultdict(set), {}
    lineage = []
    for index, path in enumerate(archives, 1):
        before = path.stat()
        archive_counts = Counter()
        fingerprint = _sha256(path)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                archive_counts["members"] += 1
                if not member.filename.endswith(".json"):
                    archive_counts["non_json_members"] += 1
                    with archive.open(member) as stream:
                        while stream.read(1024 * 1024):
                            pass
                    continue
                archive_counts["json_members"] += 1
                if member.file_size > MAX_JSON_BYTES:
                    raise ValueError(f"JSON member exceeds {MAX_JSON_BYTES} byte budget: {path.name}::{member.filename}")
                payload = archive.read(member)
                try:
                    document = json.loads(payload.decode("utf-8"))
                except UnicodeDecodeError:
                    skipped["invalid_utf8"] += 1
                    continue
                except json.JSONDecodeError:
                    skipped["invalid_json"] += 1
                    continue
                try:
                    record = parse_record(document)
                except (AttributeError, TypeError):
                    skipped["record_parse_error"] += 1
                    continue
                if record["errorCode"] is None:
                    skipped["missing_error_code"] += 1
                    continue
                if record["errorCode"] == "NOT_ASSIGNED":
                    skipped["not_assigned_error_code"] += 1
                    continue
                archive_counts["label_kept_records"] += 1
                for field, mapping in (("deviceType", "devtype_map"), ("mainState", "mainstate_map"),
                                       ("errorCode", "err_map")):
                    value = record[field]
                    if not isinstance(value, str) or not value.strip() or value != value.strip():
                        invalid_categories[field] += 1
                    elif encodings is not None and value not in encodings[mapping]:
                        unknown_categories[field][value] += 1
                code, asset = record["errorCode"], record["deviceId"]
                if isinstance(code, str):
                    labels[code] += 1
                for field, reason in _numeric_issues(document).items():
                    numeric[f"{field}:{reason}"] += 1
                if document.get("siteData", {}).get("crowd") not in CROWD:
                    counts["crowd_defaults_to_middle"] += 1
                if not _asset_valid(asset):
                    counts["invalid_asset_records"] += 1
                    continue
                assets.add(asset)
                if isinstance(code, str):
                    classes_by_asset[code].add(asset)
                try:
                    basis, timestamp = _clock(record["createdAt"])
                except (ValueError, TypeError, OverflowError):
                    counts["invalid_timestamp_records"] += 1
                    continue
                clocks[asset].add(basis)
                key = (asset, basis)
                if key in last and timestamp < last[key]:
                    counts["out_of_order_arrival_steps"] += 1
                last[key] = timestamp
                database.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?)",
                                 (split, asset, basis, timestamp, code if isinstance(code, str) else None))
                counts["timestamped_records"] += 1
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"source changed during audit: {path}")
        counts.update(archive_counts)
        lineage.append({"archive": path.name, "bytes": before.st_size, "sha256": fingerprint,
                        "crc_checked_all_members": True, **dict(archive_counts)})
        database.commit()
        print(f"[{split}] {index}/{len(archives)} ZIP; {counts['json_members']} JSON", file=sys.stderr, flush=True)
    duplicates = database.execute("""SELECT COUNT(*), COALESCE(SUM(n - 1), 0),
        COALESCE(SUM(conflict), 0) FROM (SELECT COUNT(*) n, COUNT(DISTINCT label) > 1 conflict
        FROM observations WHERE split = ? GROUP BY asset, basis, timestamp HAVING COUNT(*) > 1)""", (split,)).fetchone()
    gaps, max_gap, previous = Counter(), 0, None
    for asset, basis, stamp in database.execute(
            "SELECT asset, basis, timestamp FROM observations WHERE split = ? ORDER BY asset, basis, timestamp", (split,)):
        if previous is not None and previous[:2] == (asset, basis):
            delta = stamp - previous[2]
            bucket = "zero" if delta == 0 else "up_to_1s" if delta <= 10**9 else "up_to_10s" if delta <= 10**10 else "up_to_60s" if delta <= 6 * 10**10 else "over_60s"
            gaps[bucket] += 1
            max_gap = max(max_gap, delta)
        previous = (asset, basis, stamp)
    per_class = {code: {"records": labels[code], "assets": len(classes_by_asset[code])}
                 for code in sorted(set(labels) | (set(encodings["err_map"]) if encodings else set()))}
    return {"archive_count": len(archives), "counts": dict(counts), "skipped_json": dict(skipped),
            "per_class": per_class, "assets": sorted(assets), "archives": lineage,
            "invalid_categories": dict(invalid_categories),
            "unknown_frozen_categories": {field: dict(values) for field, values in unknown_categories.items()},
            "numeric_input_issues": dict(numeric),
            "clock": {"mixed_clock_assets": sorted(asset for asset, kinds in clocks.items() if len(kinds) > 1),
                      "duplicate_timestamp_groups": duplicates[0], "duplicate_extra_records": duplicates[1],
                      "conflicting_label_timestamp_groups": duplicates[2],
                      "sorted_adjacent_gap_counts": dict(gaps), "max_gap_seconds": max_gap / 10**9}}


def audit_source(base: Path, encoding_meta: Path | None = None) -> dict:
    start = time.monotonic()
    encodings, encoding_audit = load_encoding_metadata(encoding_meta) if encoding_meta else (None, None)
    with tempfile.TemporaryDirectory(prefix="diagnosis-source-audit-") as temporary:
        with closing(sqlite3.connect(str(Path(temporary) / "timestamps.sqlite"))) as database:
            database.execute("PRAGMA temp_store=FILE")
            database.execute("PRAGMA cache_size=-8192")
            database.execute("CREATE TABLE observations (split TEXT, asset TEXT, basis INTEGER, timestamp INTEGER, label TEXT)")
            database.execute("CREATE INDEX observation_clock ON observations (split, asset, basis, timestamp)")
            splits = {split: _scan_split(base, split, database, encodings) for split in SPLITS}
            overlap = sorted(set(splits["train"]["assets"]) & set(splits["validation"]["assets"]))
            same_times = database.execute("""SELECT COUNT(*) FROM (
                SELECT asset, basis, timestamp FROM observations GROUP BY asset, basis, timestamp
                HAVING COUNT(DISTINCT split) = 2)""").fetchone()[0]
    blockers = []
    if overlap:
        blockers.append({"code": "official_asset_overlap", "count": len(overlap)})
    for split, result in splits.items():
        if not result["counts"]["label_kept_records"]:
            blockers.append({"code": "no_label_kept_records", "split": split})
        for key in ("invalid_asset_records", "invalid_timestamp_records"):
            if result["counts"].get(key, 0):
                blockers.append({"code": key, "split": split, "count": result["counts"][key]})
        for key in ("duplicate_timestamp_groups", "mixed_clock_assets"):
            if result["clock"][key]:
                blockers.append({"code": key, "split": split})
        if any(result["skipped_json"][key] for key in INGEST_SKIP_REASONS[:3]):
            blockers.append({"code": "malformed_records_under_strict_build", "split": split})
        if result["invalid_categories"] or result["unknown_frozen_categories"]:
            blockers.append({"code": "invalid_or_unknown_categories", "split": split})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"schema": "servicerobot.diagnosis.source_audit.v1", "base_dir": str(base.resolve()),
            "audited_at_utc": datetime.now(timezone.utc).isoformat(), "encoding_audit": encoding_audit,
            "scope": "label_zip_records_only_no_windows", "splits": splits,
            "official_asset_overlap": {"count": len(overlap), "assets": overlap},
            "shared_asset_timestamp_groups": same_times,
            "current_build_blockers": blockers, "training_readiness": "not_evaluated",
            "model_performance_evaluated": False, "windows_created": False,
            "resources": {"elapsed_seconds": round(time.monotonic() - start, 3),
                          "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024},
            "limitations": ["raw_record_counts_are_not_window_support_or_independent_fault_events",
                            "source_authenticity_and_atomic_input_snapshot_not_verified",
                            "no_window_continuity_threshold_or_holdout_feasibility_decision",
                            "numeric_defaults_reported_not_changed; sensor_schema_not_fully_validated",
                            "non_json_members_crc_checked_but_not_semantically_validated",
                            "sqlite_disk_and_one_zip_directory_and_one_json_member_used",
                            "peak_rss_includes_python_imports_and_prior_process_work"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--encoding-meta", type=Path)
    parser.add_argument("--report", required=True, type=Path, help="new JSON file; never overwrite")
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        parser.exit(2, "Source audit: report must be a new file\n")
    try:
        report = audit_source(args.base_dir, args.encoding_meta)
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile, sqlite3.Error) as error:
        parser.exit(2, f"Source audit failed: {error}\n")
    print(json.dumps({"report": str(args.report), "current_build_blockers": report["current_build_blockers"],
                      "model_performance_evaluated": False}, ensure_ascii=False))
    return 2 if report["current_build_blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
