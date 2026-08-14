import json
import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


def _snapshot():
    return {
        "agvs": [
            {
                "id": "AGV-11",
                "x": 12.3,
                "y": 45.6,
                "ang": 90.0,
                "floor": 1,
                "status": "warn",
                "pred": "E-RBT-S",
                "label": "센서 이상",
                "conf": 0.88,
                "level": "위험",
                "trend_dir": "악화",
                "health": 31,
                "advice": "정비 권장",
                "sensors": {"vib": 8.2, "batt": 62.0, "temp": 49.5},
                "inference_mode": "live_booster",
                "model_latency_ms": 0.7,
                "replay_pred": "E-RBT-S",
            }
        ]
    }


class FakeMqttClient:
    def __init__(self):
        self.connected = None
        self.started = False
        self.stopped = False
        self.disconnected = False
        self.auth = None
        self.published = []

    def username_pw_set(self, username, password):
        self.auth = (username, password)

    def connect(self, host, port, keepalive):
        self.connected = (host, port, keepalive)

    def loop_start(self):
        self.started = True

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def loop_stop(self):
        self.stopped = True

    def disconnect(self):
        self.disconnected = True


def test_mqtt_bridge_builds_publish_events():
    from mqtt_bridge import build_publish_events

    events = build_publish_events(_snapshot())

    assert len(events) == 1
    event = events[0]
    assert event["topic"] == "factory/demo-fab/floor/1/agv/AGV-11/telemetry"
    assert event["validation"] == {"ok": True, "issues": []}
    assert event["payload"]["diagnosis"]["fault"] == "E-RBT-S"


def test_mqtt_bridge_dry_run_needs_no_broker():
    from mqtt_bridge import MqttBridgeConfig, build_publish_events, publish_events

    cfg = MqttBridgeConfig(host="broker.local", dry_run=True)
    result = publish_events(build_publish_events(_snapshot()), cfg)

    assert result["mode"] == "dry_run"
    assert result["broker"] == "broker.local:1883"
    assert result["published"] == 1
    assert result["invalid"] == 0


def test_mqtt_bridge_publishes_to_injected_client():
    from mqtt_bridge import MqttBridgeConfig, build_publish_events, publish_events

    client = FakeMqttClient()
    cfg = MqttBridgeConfig(
        host="mqtt.internal",
        port=1884,
        username="u",
        password="p",
        qos=1,
        retain=False,
    )
    result = publish_events(build_publish_events(_snapshot()), cfg, client=client)

    assert result["mode"] == "mqtt"
    assert client.auth == ("u", "p")
    assert client.connected == ("mqtt.internal", 1884, 30)
    assert client.started and client.stopped and client.disconnected
    assert len(client.published) == 1
    topic, payload, qos, retain = client.published[0]
    assert topic.endswith("/AGV-11/telemetry")
    assert qos == 1 and retain is False
    assert json.loads(payload)["asset_id"] == "AGV-11"


def test_mqtt_bridge_config_from_env(monkeypatch):
    from mqtt_bridge import MqttBridgeConfig

    monkeypatch.setenv("MQTT_HOST", "mqtt.example")
    monkeypatch.setenv("MQTT_PORT", "1885")
    monkeypatch.setenv("MQTT_DRY_RUN", "true")
    monkeypatch.setenv("MQTT_RETAIN", "1")

    cfg = MqttBridgeConfig.from_env()

    assert cfg.host == "mqtt.example"
    assert cfg.port == 1885
    assert cfg.dry_run is True
    assert cfg.retain is True
