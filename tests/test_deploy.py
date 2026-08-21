from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_compose_contract():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "twin-api:" in compose
    assert "dockerfile: Dockerfile" in compose
    assert 'TELEMETRY_DB: /app/data/runtime/telemetry.db' in compose
    assert '"8000:8000"' in compose
    assert "telemetry-data:/app/data/runtime" in compose
    assert "/api/snapshot" in compose
    assert "restart: unless-stopped" in compose
    assert "mqtt-broker:" in compose
    assert "eclipse-mosquitto:2" in compose
    assert 'profiles: ["mqtt", "mqtt-smoke"]' in compose
    assert "./deploy/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro" in compose
    assert "mqtt-subscriber:" in compose
    assert "mqtt_subscriber.py" in compose
    assert "http://twin-api:8000/api/edge-ingest" in compose
    assert "mqtt-smoke-publisher:" in compose
    assert "mqtt_bridge.py" in compose
    assert "http://twin-api:8000/api/snapshot" in compose


def test_dockerfile_contract_for_compose():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY data/processed/ data/processed/" in dockerfile
    assert "COPY src/ src/" in dockerfile
    assert 'CMD ["uvicorn", "realtime_server:app"' in dockerfile


def test_mosquitto_config_allows_local_smoke_profile():
    cfg = (ROOT / "deploy" / "mosquitto.conf").read_text(encoding="utf-8")

    assert "listener 1883 0.0.0.0" in cfg
    assert "allow_anonymous true" in cfg
    assert "persistence false" in cfg
