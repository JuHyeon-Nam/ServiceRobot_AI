"""
edge_gateway.py — MQTT-style edge telemetry contract for the digital twin.

The demo does not need a real broker to prove the architecture. This module
turns live AGV snapshots into topic + payload envelopes that can be sent to
MQTT later, while tests lock down the schema now.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy


SCHEMA_VERSION = "fab.edge.telemetry.v1"
SITE_ID = "demo-fab"
LINE_ID = "service-robot-ai"
REQUIRED_SENSOR_KEYS = ("vib", "batt", "temp")


def topic_for(floor: int, agv_id: str) -> str:
    return f"factory/{SITE_ID}/floor/{floor}/agv/{agv_id}/telemetry"


def edge_contract() -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "transport": "mqtt-compatible-json",
        "topic_pattern": f"factory/{SITE_ID}/floor/{{floor}}/agv/{{agv_id}}/telemetry",
        "qos": 1,
        "payload_required": [
            "schema", "ts", "site", "line", "asset_id", "floor",
            "position", "sensors", "diagnosis", "health", "source",
        ],
        "sensor_fields": list(REQUIRED_SENSOR_KEYS),
        "status_values": ["ok", "warn"],
        "consumer_path": "/api/edge-events",
        "publisher_bridge": "src/mqtt_bridge.py",
        "scale_up_path": "Run mqtt_bridge.py against a broker, then add an external subscriber / time-series sink.",
    }


def payload_from_agv(ts: float, agv: dict) -> dict:
    sensors = agv.get("sensors") or {}
    return {
        "schema": SCHEMA_VERSION,
        "ts": round(float(ts), 3),
        "site": SITE_ID,
        "line": LINE_ID,
        "asset_id": agv["id"],
        "floor": int(agv.get("floor", 0)),
        "position": {
            "x": agv.get("x"),
            "y": agv.get("y"),
            "heading_deg": agv.get("ang"),
        },
        "sensors": {k: sensors.get(k) for k in REQUIRED_SENSOR_KEYS},
        "diagnosis": {
            "status": agv.get("status"),
            "fault": agv.get("pred"),
            "label": agv.get("label"),
            "confidence": agv.get("conf"),
            "level": agv.get("level"),
            "trend": agv.get("trend_dir"),
        },
        "health": {
            "index": agv.get("health"),
            "advice": agv.get("advice"),
        },
        "source": {
            "inference_mode": agv.get("inference_mode", "live_booster"),
            "latency_ms": agv.get("model_latency_ms"),
            "replay_fault": agv.get("replay_pred"),
        },
    }


def validate_payload(payload: dict) -> list[str]:
    issues: list[str] = []
    for key in edge_contract()["payload_required"]:
        if key not in payload:
            issues.append(f"missing:{key}")
    if payload.get("schema") != SCHEMA_VERSION:
        issues.append("schema_mismatch")
    diagnosis = payload.get("diagnosis") or {}
    if diagnosis.get("status") not in {"ok", "warn"}:
        issues.append("invalid_status")
    sensors = payload.get("sensors") or {}
    for key in REQUIRED_SENSOR_KEYS:
        val = sensors.get(key)
        if not isinstance(val, (int, float)):
            issues.append(f"invalid_sensor:{key}")
    health = (payload.get("health") or {}).get("index")
    if not isinstance(health, int) or not 0 <= health <= 100:
        issues.append("invalid_health")
    return issues


class EdgeGateway:
    """In-process MQTT-style gateway buffer for live AGV telemetry envelopes."""

    def __init__(self, max_messages: int = 1000):
        self.events = deque(maxlen=max_messages)
        self.total_messages = 0
        self.invalid_messages = 0
        self.topic_counts: Counter[str] = Counter()
        self.last_ts: float | None = None

    def publish_snapshot(self, ts: float, agvs: list[dict]) -> int:
        accepted = 0
        for agv in agvs:
            payload = payload_from_agv(ts, agv)
            issues = validate_payload(payload)
            event = {
                "topic": topic_for(payload["floor"], payload["asset_id"]),
                "qos": 1,
                "retain": False,
                "payload": payload,
                "validation": {"ok": not issues, "issues": issues},
            }
            if issues:
                self.invalid_messages += 1
            else:
                accepted += 1
            self.events.append(event)
            self.total_messages += 1
            self.topic_counts[event["topic"]] += 1
            self.last_ts = payload["ts"]
        return accepted

    def recent(self, limit: int = 50, topic_prefix: str | None = None) -> list[dict]:
        rows = list(reversed(self.events))
        if topic_prefix:
            rows = [e for e in rows if e["topic"].startswith(topic_prefix)]
        return deepcopy(rows[:limit])

    def summary(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "total_messages": self.total_messages,
            "buffered_messages": len(self.events),
            "active_topics": len(self.topic_counts),
            "invalid_messages": self.invalid_messages,
            "last_ts": self.last_ts,
        }
