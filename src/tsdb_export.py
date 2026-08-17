"""
tsdb_export.py — export telemetry events to external time-series DB formats.

The runtime keeps SQLite as the compact default store. This module provides the
scale-up boundary: the same stored events can be exported as InfluxDB line
protocol or TimescaleDB SQL without adding DB client dependencies.
"""
from __future__ import annotations

import math


SCHEMA_VERSION = "fab.telemetry.tsdb_export.v1"
MEASUREMENT = "robot_pdm_events"


def tsdb_contract() -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "source": "TelemetryStore.events",
        "default_store": "sqlite",
        "supported_formats": ["json", "influx", "timescale"],
        "influx": {
            "line_protocol_endpoint": "/api/tsdb-export?fmt=influx",
            "measurement": MEASUREMENT,
            "tags": ["site", "line", "agv", "floor", "pred", "level"],
            "fields": ["conf", "health", "vib", "batt", "temp"],
            "timestamp": "event ts in nanoseconds",
        },
        "timescale": {
            "sql_endpoint": "/api/tsdb-export?fmt=timescale",
            "table": MEASUREMENT,
            "time_column": "ts",
            "hypertable": True,
        },
        "scale_up_path": "Feed MQTT bridge output or /api/tsdb-export into InfluxDB/TimescaleDB.",
    }


def _tag(v) -> str:
    s = "none" if v is None else str(v)
    return s.replace("\\", "\\\\").replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def _field_string(v) -> str:
    s = "" if v is None else str(v)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _num(v, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def to_influx_lines(rows: list[dict], site: str = "demo-fab", line: str = "service-robot-ai") -> str:
    out = []
    for r in rows:
        tags = ",".join([
            f"site={_tag(site)}",
            f"line={_tag(line)}",
            f"agv={_tag(r.get('agv'))}",
            f"floor={_tag(r.get('floor'))}",
            f"pred={_tag(r.get('pred'))}",
            f"level={_tag(r.get('level'))}",
        ])
        fields = ",".join([
            f"conf={_num(r.get('conf'))}",
            f"health={int(_num(r.get('health')))}i",
            f"vib={_num(r.get('vib'))}",
            f"batt={_num(r.get('batt'))}",
            f"temp={_num(r.get('temp'))}",
        ])
        ts_ns = int(_num(r.get("ts")) * 1_000_000_000)
        out.append(f"{MEASUREMENT},{tags} {fields} {ts_ns}")
    return "\n".join(out) + ("\n" if out else "")


def timescale_schema() -> str:
    return """CREATE TABLE IF NOT EXISTS robot_pdm_events (
  ts TIMESTAMPTZ NOT NULL,
  site TEXT NOT NULL,
  line TEXT NOT NULL,
  agv TEXT NOT NULL,
  floor INTEGER NOT NULL,
  pred TEXT NOT NULL,
  level TEXT,
  conf DOUBLE PRECISION,
  health INTEGER,
  vib DOUBLE PRECISION,
  batt DOUBLE PRECISION,
  temp DOUBLE PRECISION
);
SELECT create_hypertable('robot_pdm_events', 'ts', if_not_exists => TRUE);
"""


def _sql(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(_num(v))
    return "'" + str(v).replace("'", "''") + "'"


def to_timescale_sql(rows: list[dict], site: str = "demo-fab", line: str = "service-robot-ai") -> str:
    if not rows:
        return timescale_schema()
    values = []
    for r in rows:
        values.append("(" + ",".join([
            f"to_timestamp({_num(r.get('ts'))})",
            _sql(site),
            _sql(line),
            _sql(r.get("agv")),
            str(int(_num(r.get("floor")))),
            _sql(r.get("pred")),
            _sql(r.get("level")),
            str(_num(r.get("conf"))),
            str(int(_num(r.get("health")))),
            str(_num(r.get("vib"))),
            str(_num(r.get("batt"))),
            str(_num(r.get("temp"))),
        ]) + ")")
    return timescale_schema() + "\nINSERT INTO robot_pdm_events " \
        "(ts,site,line,agv,floor,pred,level,conf,health,vib,batt,temp) VALUES\n" \
        + ",\n".join(values) + "\nON CONFLICT DO NOTHING;\n"


def export_payload(rows: list[dict], fmt: str) -> tuple[str, str]:
    if fmt == "json":
        import json

        body = {
            "schema": SCHEMA_VERSION,
            "contract": tsdb_contract(),
            "events": rows,
        }
        return json.dumps(body, ensure_ascii=False), "application/json; charset=utf-8"
    if fmt == "influx":
        return to_influx_lines(rows), "text/plain; charset=utf-8"
    if fmt == "timescale":
        return to_timescale_sql(rows), "application/sql; charset=utf-8"
    raise ValueError(f"unsupported tsdb export format: {fmt}")
