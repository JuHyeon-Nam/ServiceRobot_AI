"""
physical_sensor_adapter.py — physical/edge sensor adapter example.

This adapter converts simple line-delimited sensor records into the standard
fab.edge.telemetry.v1 payload and forwards them to /api/edge-ingest.

It intentionally keeps the hardware boundary simple:
- JSON line: {"asset_id":"AGV-01","floor":0,"vib":6.2,"batt":42,"temp":61}
- CSV line:  AGV-01,0,6.2,42,61,120,80,90

Dry-run mode prints the normalized payload without posting:
    printf '{"asset_id":"AGV-01","floor":0,"vib":6.2,"batt":42,"temp":61}\n' \
      | python src/physical_sensor_adapter.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, TextIO

from edge_gateway import LINE_ID, SCHEMA_VERSION, SITE_ID, topic_for, validate_payload


DEFAULT_INGEST_URL = "http://127.0.0.1:8000/api/edge-ingest"
CSV_FIELDS = ("asset_id", "floor", "vib", "batt", "temp", "x", "y", "heading_deg")


@dataclass(frozen=True)
class SensorAdapterConfig:
    ingest_url: str = DEFAULT_INGEST_URL
    dry_run: bool = False
    default_asset: str = "AGV-EDGE-01"
    default_floor: int = 0
    source_name: str = "physical_sensor_adapter"
    interval_sec: float = 0.0

    @classmethod
    def from_env(cls) -> "SensorAdapterConfig":
        return cls(
            ingest_url=os.environ.get("EDGE_INGEST_URL", DEFAULT_INGEST_URL),
            dry_run=os.environ.get("EDGE_DRY_RUN", "0").lower() in {"1", "true", "yes"},
            default_asset=os.environ.get("EDGE_ASSET_ID", "AGV-EDGE-01"),
            default_floor=int(os.environ.get("EDGE_FLOOR", "0")),
            source_name=os.environ.get("EDGE_SOURCE_NAME", "physical_sensor_adapter"),
            interval_sec=float(os.environ.get("EDGE_INTERVAL_SEC", "0")),
        )


def _num(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def parse_sensor_line(line: str, cfg: SensorAdapterConfig | None = None) -> dict:
    """Parse one JSON or CSV sensor record into a normalized dict."""
    cfg = cfg or SensorAdapterConfig()
    raw = line.strip()
    if not raw:
        raise ValueError("empty_line")
    if raw.startswith("{"):
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError("json_not_object")
    else:
        values = next(csv.reader([raw]))
        row = {k: values[i] for i, k in enumerate(CSV_FIELDS) if i < len(values)}

    return {
        "asset_id": str(row.get("asset_id") or cfg.default_asset),
        "floor": _int(row.get("floor"), cfg.default_floor),
        "vib": _num(row.get("vib")),
        "batt": _num(row.get("batt"), 100.0),
        "temp": _num(row.get("temp"), 25.0),
        "x": _num(row.get("x"), 0.0),
        "y": _num(row.get("y"), 0.0),
        "heading_deg": _num(row.get("heading_deg", row.get("heading")), 0.0),
        "status": row.get("status"),
        "fault": row.get("fault"),
        "confidence": row.get("confidence"),
        "health": row.get("health"),
    }


def infer_diagnosis(record: dict) -> dict:
    """Small deterministic safety triage for raw physical records."""
    vib = float(record["vib"])
    batt = float(record["batt"])
    temp = float(record["temp"])
    explicit_fault = record.get("fault")
    explicit_status = record.get("status")
    if explicit_fault:
        status = explicit_status or ("ok" if explicit_fault == "정상" else "warn")
        fault = explicit_fault
    elif batt < 25:
        status, fault = "warn", "E-RBT-B"
    elif vib > 5.0:
        status, fault = "warn", "E-RBT-S"
    elif temp > 58.0:
        status, fault = "warn", "E-RBT-E"
    else:
        status, fault = "ok", "정상"

    confidence = record.get("confidence")
    conf = float(confidence) if confidence not in (None, "") else (0.92 if status == "warn" else 0.98)
    level = None
    if status == "warn":
        level = "위험" if conf >= 0.85 or batt < 20 or vib > 7 or temp > 65 else "경고"
    label = {
        "E-RBT-B": "배터리 저하",
        "E-RBT-S": "센서 이상",
        "E-RBT-E": "긴급정지",
        "정상": "정상",
    }.get(fault, fault)
    return {"status": status, "fault": fault, "label": label, "confidence": round(conf, 3), "level": level}


def health_index(record: dict, diagnosis: dict) -> int:
    explicit = record.get("health")
    if explicit not in (None, ""):
        return max(0, min(100, _int(explicit, 100)))
    score = 100
    score -= max(0.0, float(record["vib"]) - 3.0) * 9
    score -= max(0.0, 35.0 - float(record["batt"])) * 1.4
    score -= max(0.0, float(record["temp"]) - 45.0) * 1.1
    if diagnosis["status"] == "warn":
        score -= 12
    return int(max(2, min(100, round(score))))


def payload_from_sensor_record(record: dict, cfg: SensorAdapterConfig | None = None, ts: float | None = None) -> dict:
    cfg = cfg or SensorAdapterConfig()
    ts = time.time() if ts is None else ts
    diagnosis = infer_diagnosis(record)
    health = health_index(record, diagnosis)
    return {
        "schema": SCHEMA_VERSION,
        "ts": round(float(ts), 3),
        "site": SITE_ID,
        "line": LINE_ID,
        "asset_id": record["asset_id"],
        "floor": int(record["floor"]),
        "position": {"x": record["x"], "y": record["y"], "heading_deg": record["heading_deg"]},
        "sensors": {"vib": record["vib"], "batt": record["batt"], "temp": record["temp"]},
        "diagnosis": {**diagnosis, "trend": "악화" if diagnosis["status"] == "warn" else "안정"},
        "health": {"index": health, "advice": "정비 필요" if health < 55 else "정상 운전"},
        "source": {
            "inference_mode": cfg.source_name,
            "latency_ms": 0,
            "replay_fault": None,
        },
    }


def post_payload(payload: dict, cfg: SensorAdapterConfig, http_post: Any = None) -> dict:
    topic = topic_for(payload["floor"], payload["asset_id"])
    issues = validate_payload(payload)
    if issues:
        return {"accepted": False, "topic": topic, "validation": {"ok": False, "issues": issues}}
    if cfg.dry_run:
        return {"accepted": True, "mode": "dry_run", "topic": topic, "payload": payload}
    if http_post is None:
        import requests

        http_post = requests.post
    res = http_post(cfg.ingest_url, params={"topic": topic}, json=payload, timeout=5)
    res.raise_for_status()
    return {"accepted": True, "mode": "http_ingest", "topic": topic, "response": res.json()}


def iter_lines(handle: TextIO) -> Iterable[str]:
    for line in handle:
        if line.strip():
            yield line


def run_stream(handle: TextIO, cfg: SensorAdapterConfig, http_post: Any = None) -> list[dict]:
    results = []
    for line in iter_lines(handle):
        record = parse_sensor_line(line, cfg)
        payload = payload_from_sensor_record(record, cfg)
        result = post_payload(payload, cfg, http_post=http_post)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
        if cfg.interval_sec > 0:
            time.sleep(cfg.interval_sec)
    return results


def main(argv: list[str] | None = None) -> int:
    env = SensorAdapterConfig.from_env()
    p = argparse.ArgumentParser(description="Convert physical sensor lines to /api/edge-ingest payloads.")
    p.add_argument("--ingest-url", default=env.ingest_url)
    p.add_argument("--asset-id", default=env.default_asset)
    p.add_argument("--floor", type=int, default=env.default_floor)
    p.add_argument("--source-name", default=env.source_name)
    p.add_argument("--interval", type=float, default=env.interval_sec)
    p.add_argument("--input", help="Input file path. Defaults to stdin.")
    p.add_argument("--dry-run", action="store_true", default=env.dry_run)
    args = p.parse_args(argv)

    cfg = SensorAdapterConfig(
        ingest_url=args.ingest_url,
        dry_run=args.dry_run,
        default_asset=args.asset_id,
        default_floor=args.floor,
        source_name=args.source_name,
        interval_sec=args.interval,
    )
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            run_stream(f, cfg)
    else:
        run_stream(sys.stdin, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
