import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


def _agv(**overrides):
    row = {
        "id": "AGV-01",
        "x": 12.3,
        "y": 45.6,
        "ang": 90.0,
        "floor": 2,
        "status": "warn",
        "pred": "E-RBT-B",
        "label": "배터리 저하",
        "conf": 0.91,
        "level": "위험",
        "trend_dir": "악화",
        "health": 24,
        "advice": "정비 필요 · 우선 대응",
        "sensors": {"vib": 4.1, "batt": 19.5, "temp": 51.2},
        "inference_mode": "live_booster",
        "model_latency_ms": 0.8,
        "replay_pred": "E-RBT-B",
    }
    row.update(overrides)
    return row


def test_edge_topic_and_payload_contract():
    from edge_gateway import SCHEMA_VERSION, payload_from_agv, topic_for, validate_payload

    payload = payload_from_agv(100.1234, _agv())

    assert topic_for(2, "AGV-01") == "factory/demo-fab/floor/2/agv/AGV-01/telemetry"
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["asset_id"] == "AGV-01"
    assert payload["diagnosis"]["fault"] == "E-RBT-B"
    assert payload["sensors"] == {"vib": 4.1, "batt": 19.5, "temp": 51.2}
    assert validate_payload(payload) == []


def test_edge_payload_validation_catches_schema_breaks():
    from edge_gateway import payload_from_agv, validate_payload

    payload = payload_from_agv(100.0, _agv())
    payload["sensors"]["batt"] = None
    payload["health"]["index"] = 120
    payload["diagnosis"]["status"] = "unknown"

    issues = validate_payload(payload)
    assert "invalid_sensor:batt" in issues
    assert "invalid_health" in issues
    assert "invalid_status" in issues


def test_edge_gateway_buffers_recent_events_and_summary():
    from edge_gateway import EdgeGateway

    gateway = EdgeGateway(max_messages=2)
    assert gateway.publish_snapshot(100.0, [_agv(id="AGV-01"), _agv(id="AGV-02")]) == 2
    gateway.publish_snapshot(101.0, [_agv(id="AGV-03")])

    summary = gateway.summary()
    assert summary["total_messages"] == 3
    assert summary["buffered_messages"] == 2
    assert summary["active_topics"] == 3
    assert summary["invalid_messages"] == 0

    recent = gateway.recent(limit=10)
    assert [e["payload"]["asset_id"] for e in recent] == ["AGV-03", "AGV-02"]
    assert gateway.recent(topic_prefix="factory/demo-fab/floor/2/agv/AGV-03")[0]["topic"].endswith("/telemetry")


def test_edge_gateway_ingests_inbound_payload():
    from edge_gateway import EdgeGateway, payload_from_agv

    gateway = EdgeGateway(max_messages=4)
    payload = payload_from_agv(200.0, _agv(id="AGV-04"))
    event = gateway.ingest_payload(payload)

    assert event["direction"] == "inbound"
    assert event["validation"]["ok"] is True
    summary = gateway.summary()
    assert summary["ingested_messages"] == 1
    assert summary["total_messages"] == 1
    assert gateway.recent()[0]["payload"]["asset_id"] == "AGV-04"


def test_edge_contract_document_is_machine_readable():
    from edge_gateway import edge_contract

    contract = edge_contract()
    assert contract["transport"] == "mqtt-compatible-json"
    assert "{floor}" in contract["topic_pattern"] and "{agv_id}" in contract["topic_pattern"]
    assert "sensors" in contract["payload_required"]
    assert contract["qos"] == 1
    assert contract["ingest_path"] == "/api/edge-ingest"
    assert contract["publisher_bridge"] == "src/mqtt_bridge.py"
    assert contract["subscriber_bridge"] == "src/mqtt_subscriber.py"
