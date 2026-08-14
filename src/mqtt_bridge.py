"""
mqtt_bridge.py — optional real MQTT publisher for the digital twin telemetry.

The realtime server already exposes a broker-compatible edge contract. This
module is the replaceable bridge that reads /api/snapshot and publishes each
AGV telemetry envelope to a real MQTT broker when paho-mqtt is installed.

Dry-run mode needs no broker and no paho dependency:
    python src/mqtt_bridge.py --once --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from edge_gateway import payload_from_agv, topic_for, validate_payload


DEFAULT_SNAPSHOT_URL = "http://127.0.0.1:8000/api/snapshot"


@dataclass(frozen=True)
class MqttBridgeConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    client_id: str = "service-robot-ai-bridge"
    username: str | None = None
    password: str | None = None
    qos: int = 1
    retain: bool = False
    keepalive: int = 30
    snapshot_url: str = DEFAULT_SNAPSHOT_URL
    interval_sec: float = 1.0
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "MqttBridgeConfig":
        return cls(
            host=os.environ.get("MQTT_HOST", "127.0.0.1"),
            port=int(os.environ.get("MQTT_PORT", "1883")),
            client_id=os.environ.get("MQTT_CLIENT_ID", "service-robot-ai-bridge"),
            username=os.environ.get("MQTT_USERNAME") or None,
            password=os.environ.get("MQTT_PASSWORD") or None,
            qos=int(os.environ.get("MQTT_QOS", "1")),
            retain=os.environ.get("MQTT_RETAIN", "0").lower() in {"1", "true", "yes"},
            keepalive=int(os.environ.get("MQTT_KEEPALIVE", "30")),
            snapshot_url=os.environ.get("MQTT_SNAPSHOT_URL", DEFAULT_SNAPSHOT_URL),
            interval_sec=float(os.environ.get("MQTT_INTERVAL_SEC", "1.0")),
            dry_run=os.environ.get("MQTT_DRY_RUN", "0").lower() in {"1", "true", "yes"},
        )


def fetch_snapshot(url: str = DEFAULT_SNAPSHOT_URL) -> dict:
    """Fetch the realtime server snapshot. Kept separate for testability."""
    import requests

    res = requests.get(url, timeout=5)
    res.raise_for_status()
    return res.json()


def build_publish_events(snapshot: dict) -> list[dict]:
    """Convert a realtime snapshot into MQTT publish events."""
    ts = time.time()
    events: list[dict] = []
    for agv in snapshot.get("agvs", []):
        payload = payload_from_agv(ts, agv)
        issues = validate_payload(payload)
        events.append({
            "topic": topic_for(payload["floor"], payload["asset_id"]),
            "payload": payload,
            "validation": {"ok": not issues, "issues": issues},
        })
    return events


def _load_paho_client_class():
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "paho-mqtt is required for real broker publishing. "
            "Install it or run with --dry-run."
        ) from exc
    return mqtt.Client


def publish_events(events: list[dict], cfg: MqttBridgeConfig, client: Any = None) -> dict:
    """Publish events to MQTT, or validate them in dry-run mode.

    A fake client can be passed by tests. The fake only needs connect, publish,
    loop_start, loop_stop, and disconnect methods.
    """
    valid_events = [e for e in events if e["validation"]["ok"]]
    invalid = len(events) - len(valid_events)
    if cfg.dry_run:
        return {
            "mode": "dry_run",
            "broker": f"{cfg.host}:{cfg.port}",
            "published": len(valid_events),
            "invalid": invalid,
            "topics": [e["topic"] for e in valid_events],
        }

    if client is None:
        client_cls = _load_paho_client_class()
        client = client_cls(client_id=cfg.client_id)
    if cfg.username:
        client.username_pw_set(cfg.username, cfg.password)

    client.connect(cfg.host, cfg.port, cfg.keepalive)
    client.loop_start()
    published = 0
    try:
        for event in valid_events:
            payload = json.dumps(event["payload"], ensure_ascii=False, separators=(",", ":"))
            client.publish(event["topic"], payload=payload, qos=cfg.qos, retain=cfg.retain)
            published += 1
    finally:
        client.loop_stop()
        client.disconnect()

    return {
        "mode": "mqtt",
        "broker": f"{cfg.host}:{cfg.port}",
        "published": published,
        "invalid": invalid,
        "topics": [e["topic"] for e in valid_events],
    }


def run_once(cfg: MqttBridgeConfig, client: Any = None) -> dict:
    snapshot = fetch_snapshot(cfg.snapshot_url)
    return publish_events(build_publish_events(snapshot), cfg, client=client)


def main(argv: list[str] | None = None) -> int:
    env = MqttBridgeConfig.from_env()
    p = argparse.ArgumentParser(description="Publish ServiceRobot_AI telemetry to an MQTT broker.")
    p.add_argument("--host", default=env.host)
    p.add_argument("--port", type=int, default=env.port)
    p.add_argument("--client-id", default=env.client_id)
    p.add_argument("--username", default=env.username)
    p.add_argument("--password", default=env.password)
    p.add_argument("--qos", type=int, choices=(0, 1, 2), default=env.qos)
    p.add_argument("--retain", action="store_true", default=env.retain)
    p.add_argument("--snapshot-url", default=env.snapshot_url)
    p.add_argument("--interval", type=float, default=env.interval_sec)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=env.dry_run)
    args = p.parse_args(argv)

    cfg = MqttBridgeConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        username=args.username,
        password=args.password,
        qos=args.qos,
        retain=args.retain,
        snapshot_url=args.snapshot_url,
        interval_sec=args.interval,
        dry_run=args.dry_run,
    )

    while True:
        result = run_once(cfg)
        print(json.dumps(result, ensure_ascii=False))
        if args.once:
            return 0
        time.sleep(cfg.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
