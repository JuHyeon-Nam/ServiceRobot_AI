import json
import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


def _payload(**overrides):
    body = {
        "schema": "fab.edge.telemetry.v1",
        "ts": 100.0,
        "site": "demo-fab",
        "line": "service-robot-ai",
        "asset_id": "AGV-09",
        "floor": 1,
        "position": {"x": 10.0, "y": 20.0, "heading_deg": 90.0},
        "sensors": {"vib": 6.1, "batt": 44.0, "temp": 55.0},
        "diagnosis": {
            "status": "warn",
            "fault": "E-RBT-S",
            "label": "센서 이상",
            "confidence": 0.92,
            "level": "위험",
            "trend": "악화",
        },
        "health": {"index": 28, "advice": "정비 필요"},
        "source": {"inference_mode": "mqtt_ingest", "latency_ms": 0, "replay_fault": "E-RBT-S"},
    }
    body.update(overrides)
    return body


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FakeClient:
    def __init__(self):
        self.auth = None
        self.connected = None
        self.subscribed = None
        self.disconnected = False
        self.on_connect = None
        self.on_message = None

    def username_pw_set(self, username, password):
        self.auth = (username, password)

    def connect(self, host, port, keepalive):
        self.connected = (host, port, keepalive)

    def subscribe(self, topic, qos):
        self.subscribed = (topic, qos)

    def disconnect(self):
        self.disconnected = True


class FakeMsg:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload


def test_mqtt_subscriber_dry_run_validates_payload_without_http():
    from mqtt_subscriber import MqttSubscriberConfig, handle_message

    cfg = MqttSubscriberConfig(dry_run=True)
    out = handle_message("factory/demo-fab/floor/1/agv/AGV-09/telemetry", json.dumps(_payload()), cfg)

    assert out["accepted"] is True
    assert out["mode"] == "dry_run"
    assert out["asset_id"] == "AGV-09"


def test_mqtt_subscriber_posts_valid_payload_to_edge_ingest():
    from mqtt_subscriber import MqttSubscriberConfig, handle_message

    calls = []

    def fake_post(url, params, json, timeout):
        calls.append((url, params, json, timeout))
        return FakeResponse({"accepted": True, "asset_id": json["asset_id"]})

    cfg = MqttSubscriberConfig(ingest_url="http://api.local/api/edge-ingest")
    topic = "factory/demo-fab/floor/1/agv/AGV-09/telemetry"
    out = handle_message(topic, json.dumps(_payload()).encode("utf-8"), cfg, http_post=fake_post)

    assert out["accepted"] is True
    assert out["mode"] == "http_ingest"
    assert calls == [("http://api.local/api/edge-ingest", {"topic": topic}, _payload(), 5)]


def test_mqtt_subscriber_rejects_invalid_payload_before_post():
    from mqtt_subscriber import MqttSubscriberConfig, handle_message

    calls = []
    bad = _payload()
    bad["sensors"]["vib"] = None
    cfg = MqttSubscriberConfig()
    out = handle_message("topic", json.dumps(bad), cfg, http_post=lambda **kw: calls.append(kw))

    assert out["accepted"] is False
    assert "invalid_sensor:vib" in out["validation"]["issues"]
    assert calls == []


def test_mqtt_subscriber_configures_client_callbacks():
    from mqtt_subscriber import MqttSubscriberConfig, configure_client

    results = []
    posts = []
    cfg = MqttSubscriberConfig(username="u", password="p", qos=2, once=True)
    client = configure_client(
        FakeClient(),
        cfg,
        http_post=lambda url, params, json, timeout: posts.append((url, params, json, timeout))
        or FakeResponse({"accepted": True}),
        on_result=results.append,
    )

    client.on_connect(client, None, None, 0)
    assert client.auth == ("u", "p")
    assert client.subscribed == (cfg.topic, 2)

    msg = FakeMsg("factory/demo-fab/floor/1/agv/AGV-09/telemetry", json.dumps(_payload()).encode("utf-8"))
    client.on_message(client, None, msg)

    assert posts and posts[0][1] == {"topic": msg.topic}
    assert results[-1]["accepted"] is True
    assert client.disconnected is True


def test_mqtt_subscriber_config_from_env(monkeypatch):
    from mqtt_subscriber import MqttSubscriberConfig

    monkeypatch.setenv("MQTT_HOST", "broker.local")
    monkeypatch.setenv("MQTT_PORT", "1887")
    monkeypatch.setenv("MQTT_SUB_TOPIC", "factory/demo/#")
    monkeypatch.setenv("MQTT_INGEST_URL", "http://api/ingest")
    monkeypatch.setenv("MQTT_ONCE", "yes")

    cfg = MqttSubscriberConfig.from_env()

    assert cfg.host == "broker.local"
    assert cfg.port == 1887
    assert cfg.topic == "factory/demo/#"
    assert cfg.ingest_url == "http://api/ingest"
    assert cfg.once is True
