"""
mqtt_subscriber.py — inbound MQTT subscriber for edge telemetry.

This runner closes the broker-to-twin loop:
MQTT broker topic -> validated edge payload -> POST /api/edge-ingest -> WebSocket twin update.

Dry-run mode validates payload handling without requiring a broker or paho-mqtt:
    python src/mqtt_subscriber.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from edge_gateway import edge_contract, validate_payload


DEFAULT_INGEST_URL = "http://127.0.0.1:8000/api/edge-ingest"
DEFAULT_TOPIC = "factory/demo-fab/floor/+/agv/+/telemetry"


@dataclass(frozen=True)
class MqttSubscriberConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    client_id: str = "service-robot-ai-subscriber"
    username: str | None = None
    password: str | None = None
    qos: int = 1
    keepalive: int = 30
    topic: str = DEFAULT_TOPIC
    ingest_url: str = DEFAULT_INGEST_URL
    dry_run: bool = False
    once: bool = False

    @classmethod
    def from_env(cls) -> "MqttSubscriberConfig":
        return cls(
            host=os.environ.get("MQTT_HOST", "127.0.0.1"),
            port=int(os.environ.get("MQTT_PORT", "1883")),
            client_id=os.environ.get("MQTT_SUB_CLIENT_ID", "service-robot-ai-subscriber"),
            username=os.environ.get("MQTT_USERNAME") or None,
            password=os.environ.get("MQTT_PASSWORD") or None,
            qos=int(os.environ.get("MQTT_QOS", "1")),
            keepalive=int(os.environ.get("MQTT_KEEPALIVE", "30")),
            topic=os.environ.get("MQTT_SUB_TOPIC", DEFAULT_TOPIC),
            ingest_url=os.environ.get("MQTT_INGEST_URL", DEFAULT_INGEST_URL),
            dry_run=os.environ.get("MQTT_DRY_RUN", "0").lower() in {"1", "true", "yes"},
            once=os.environ.get("MQTT_ONCE", "0").lower() in {"1", "true", "yes"},
        )


def parse_payload(raw: bytes | str) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError("payload_not_object")
    return body


def ingest_payload(
    topic: str,
    payload: dict,
    cfg: MqttSubscriberConfig,
    http_post: Callable[..., Any] | None = None,
) -> dict:
    """Validate and forward one MQTT payload to the realtime server ingest API."""
    issues = validate_payload(payload)
    if issues:
        return {"accepted": False, "topic": topic, "validation": {"ok": False, "issues": issues}}
    if cfg.dry_run:
        return {"accepted": True, "mode": "dry_run", "topic": topic, "asset_id": payload.get("asset_id")}

    if http_post is None:
        import requests

        http_post = requests.post
    res = http_post(cfg.ingest_url, params={"topic": topic}, json=payload, timeout=5)
    res.raise_for_status()
    body = res.json()
    return {
        "accepted": bool(body.get("accepted")),
        "mode": "http_ingest",
        "topic": topic,
        "asset_id": payload.get("asset_id"),
        "response": body,
    }


def handle_message(
    topic: str,
    raw_payload: bytes | str,
    cfg: MqttSubscriberConfig,
    http_post: Callable[..., Any] | None = None,
) -> dict:
    try:
        payload = parse_payload(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {"accepted": False, "topic": topic, "validation": {"ok": False, "issues": [str(exc)]}}
    return ingest_payload(topic, payload, cfg, http_post=http_post)


def _load_paho_client_class():
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "paho-mqtt is required for real broker subscribing. "
            "Install it or run tests/dry-run without broker mode."
        ) from exc
    return mqtt.Client


def configure_client(
    client: Any,
    cfg: MqttSubscriberConfig,
    http_post: Callable[..., Any] | None = None,
    on_result: Callable[[dict], None] | None = None,
) -> Any:
    if cfg.username:
        client.username_pw_set(cfg.username, cfg.password)

    def on_connect(c, _userdata, _flags, rc, *args):
        if rc != 0:
            result = {"accepted": False, "error": f"mqtt_connect_failed:{rc}"}
            if on_result:
                on_result(result)
            return
        c.subscribe(cfg.topic, qos=cfg.qos)
        if on_result:
            on_result({"connected": True, "topic": cfg.topic, "qos": cfg.qos})

    def on_message(c, _userdata, msg):
        result = handle_message(msg.topic, msg.payload, cfg, http_post=http_post)
        if on_result:
            on_result(result)
        if cfg.once:
            c.disconnect()

    client.on_connect = on_connect
    client.on_message = on_message
    return client


def run_subscriber(
    cfg: MqttSubscriberConfig,
    client: Any = None,
    http_post: Callable[..., Any] | None = None,
    on_result: Callable[[dict], None] | None = None,
) -> dict:
    if cfg.dry_run:
        return {
            "mode": "dry_run",
            "broker": f"{cfg.host}:{cfg.port}",
            "topic": cfg.topic,
            "ingest_url": cfg.ingest_url,
            "contract": edge_contract()["schema"],
        }

    if client is None:
        client_cls = _load_paho_client_class()
        client = client_cls(client_id=cfg.client_id)
    configure_client(client, cfg, http_post=http_post, on_result=on_result)
    client.connect(cfg.host, cfg.port, cfg.keepalive)
    client.loop_forever()
    return {"mode": "mqtt_subscriber", "broker": f"{cfg.host}:{cfg.port}", "topic": cfg.topic}


def main(argv: list[str] | None = None) -> int:
    env = MqttSubscriberConfig.from_env()
    p = argparse.ArgumentParser(description="Subscribe MQTT edge telemetry and POST it to /api/edge-ingest.")
    p.add_argument("--host", default=env.host)
    p.add_argument("--port", type=int, default=env.port)
    p.add_argument("--client-id", default=env.client_id)
    p.add_argument("--username", default=env.username)
    p.add_argument("--password", default=env.password)
    p.add_argument("--qos", type=int, choices=(0, 1, 2), default=env.qos)
    p.add_argument("--topic", default=env.topic)
    p.add_argument("--ingest-url", default=env.ingest_url)
    p.add_argument("--once", action="store_true", default=env.once)
    p.add_argument("--dry-run", action="store_true", default=env.dry_run)
    args = p.parse_args(argv)

    cfg = MqttSubscriberConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        username=args.username,
        password=args.password,
        qos=args.qos,
        topic=args.topic,
        ingest_url=args.ingest_url,
        once=args.once,
        dry_run=args.dry_run,
    )

    def emit(result: dict) -> None:
        print(json.dumps(dict(result, ts=round(time.time(), 3)), ensure_ascii=False))

    result = run_subscriber(cfg, on_result=emit)
    emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
