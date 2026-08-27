"""
build_rul_dataset.py — build a supervised-ready RUL training table.

Inputs:
- telemetry events from TelemetryStore SQLite or CSV export
- failure labels with asset_id, failure_ts, failure_code

Output:
- one row per telemetry event with the RUL feature vector
- minutes_to_failure for rows before an observed failure
- right-censoring metadata for rows without a future failure in the observation window
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict

from rul_runtime import RUL_FEATURES, RUL_SCHEMA, RUL_TARGET, rul_feature_vector, rul_readiness_report


OUTPUT_COLUMNS = [
    "asset_id",
    "event_ts",
    "failure_ts",
    "failure_code",
    "censoring",
    "event_observed",
    "minutes_to_failure",
    "observed_duration_min",
    *RUL_FEATURES,
]


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "warn", "위험", "경고", "주의"}


def _asset_id(row: dict) -> str:
    return str(row.get("asset_id") or row.get("agv") or row.get("id") or "").strip()


def _event_ts(row: dict) -> float:
    return _num(row.get("event_ts", row.get("ts")))


def _derive_risk_score(row: dict, health: float, warn: bool) -> float:
    if row.get("risk_score") not in (None, ""):
        return _num(row.get("risk_score"))
    level = row.get("severity") or row.get("level")
    severity = {"정상": 0, "주의": 1, "경고": 2, "위험": 3}.get(level, 1 if warn else 0)
    vib_hit = 12 if _num(row.get("vib")) > 5 else 0
    batt_hit = 12 if _num(row.get("batt"), 100) < 30 else 0
    temp_hit = 10 if _num(row.get("temp")) > 58 else 0
    return max(0.0, min(100.0, (100.0 - health) + severity * 10 + vib_hit + batt_hit + temp_hit))


def normalize_event(row: dict) -> dict:
    asset_id = _asset_id(row)
    event_ts = _event_ts(row)
    pred = str(row.get("pred") or row.get("fault") or "정상")
    warn = _bool(row.get("warn", row.get("status"))) or pred != "정상" or bool(row.get("level"))
    health = _num(row.get("health"), 100.0)
    phm = {
        "risk_score": _derive_risk_score(row, health, warn),
        "trend_slope": _num(row.get("trend_slope")),
        "severity": row.get("severity") or row.get("level") or ("주의" if warn else "정상"),
    }
    sensors = {"vib": row.get("vib"), "batt": row.get("batt"), "temp": row.get("temp")}
    return {
        "asset_id": asset_id,
        "event_ts": event_ts,
        "features": rul_feature_vector(int(round(health)), phm, sensors, warn),
    }


def read_events_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [normalize_event(row) for row in csv.DictReader(f) if _asset_id(row)]


def read_events_sqlite(path: str, limit: int | None = None, since: float = 0.0) -> list[dict]:
    with sqlite3.connect(path) as cx:
        cols = {row[1] for row in cx.execute("PRAGMA table_info(events)").fetchall()}
        risk_expr = "risk_score" if "risk_score" in cols else "NULL AS risk_score"
        slope_expr = "trend_slope" if "trend_slope" in cols else "NULL AS trend_slope"
        q = (
            "SELECT ts,agv,floor,pred,conf,level,health,vib,batt,temp,"
            f"{risk_expr},{slope_expr} "
            "FROM events WHERE ts>=? ORDER BY ts ASC"
        )
        args: list = [since]
        if limit:
            q += " LIMIT ?"
            args.append(limit)
        cur = cx.execute(q, args)
        names = [c[0] for c in cur.description]
        return [normalize_event(dict(zip(names, row))) for row in cur.fetchall()]


def read_failure_labels(path: str) -> list[dict]:
    labels = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            asset_id = _asset_id(row)
            if not asset_id:
                continue
            labels.append({
                "asset_id": asset_id,
                "failure_ts": _num(row.get("failure_ts")),
                "failure_code": row.get("failure_code") or row.get("pred") or "unknown",
                "censoring": _bool(row.get("censoring")),
            })
    return sorted(labels, key=lambda r: (r["asset_id"], r["failure_ts"]))


def build_rul_rows(events: list[dict], labels: list[dict], include_censored: bool = True) -> list[dict]:
    failures_by_asset: dict[str, list[dict]] = defaultdict(list)
    for label in labels:
        if not label["censoring"] and label["failure_ts"] > 0:
            failures_by_asset[label["asset_id"]].append(label)
    last_seen = {}
    for event in events:
        last_seen[event["asset_id"]] = max(last_seen.get(event["asset_id"], 0.0), event["event_ts"])

    out = []
    for event in sorted(events, key=lambda r: (r["asset_id"], r["event_ts"])):
        next_failure = next(
            (f for f in failures_by_asset.get(event["asset_id"], []) if f["failure_ts"] >= event["event_ts"]),
            None,
        )
        if next_failure:
            duration = max(next_failure["failure_ts"] - event["event_ts"], 0.0) / 60.0
            row = {
                "asset_id": event["asset_id"],
                "event_ts": event["event_ts"],
                "failure_ts": next_failure["failure_ts"],
                "failure_code": next_failure["failure_code"],
                "censoring": False,
                "event_observed": 1,
                "minutes_to_failure": round(duration, 3),
                "observed_duration_min": round(duration, 3),
            }
        elif include_censored:
            duration = max(last_seen.get(event["asset_id"], event["event_ts"]) - event["event_ts"], 0.0) / 60.0
            row = {
                "asset_id": event["asset_id"],
                "event_ts": event["event_ts"],
                "failure_ts": "",
                "failure_code": "none",
                "censoring": True,
                "event_observed": 0,
                "minutes_to_failure": "",
                "observed_duration_min": round(duration, 3),
            }
        else:
            continue
        row.update(event["features"])
        out.append(row)
    return out


def dataset_metadata(rows: list[dict]) -> dict:
    readiness_rows = [
        {
            "asset_id": row["asset_id"],
            "event_ts": row["event_ts"],
            "failure_ts": row["failure_ts"],
            "failure_code": row["failure_code"],
            "censoring": row["censoring"],
        }
        for row in rows
    ]
    return {
        "schema": RUL_SCHEMA,
        "target": RUL_TARGET,
        "rows": len(rows),
        "failure_rows": sum(int(row["event_observed"]) == 1 for row in rows),
        "censored_rows": sum(int(row["event_observed"]) == 0 for row in rows),
        "feature_fields": list(RUL_FEATURES),
        "output_columns": OUTPUT_COLUMNS,
        "readiness": rul_readiness_report(readiness_rows),
    }


def write_csv(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build supervised-ready RUL dataset from telemetry and failure labels.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--events-csv", help="Telemetry events CSV with agv/asset_id, ts/event_ts, sensors, PHM features.")
    src.add_argument("--telemetry-db", help="TelemetryStore SQLite DB path.")
    p.add_argument("--labels-csv", required=True, help="Failure labels CSV with asset_id, failure_ts, failure_code.")
    p.add_argument("--out-csv", default="../data/processed/rul_training.csv")
    p.add_argument("--meta-json", default=None)
    p.add_argument("--since", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--drop-censored", action="store_true", help="Write only rows with observed future failure.")
    args = p.parse_args(argv)

    events = read_events_csv(args.events_csv) if args.events_csv else read_events_sqlite(
        args.telemetry_db, limit=args.limit, since=args.since
    )
    labels = read_failure_labels(args.labels_csv)
    rows = build_rul_rows(events, labels, include_censored=not args.drop_censored)
    write_csv(rows, args.out_csv)
    meta = dataset_metadata(rows)
    meta_path = args.meta_json or os.path.splitext(args.out_csv)[0] + ".metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
