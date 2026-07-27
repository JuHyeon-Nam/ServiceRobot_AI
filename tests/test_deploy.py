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


def test_dockerfile_contract_for_compose():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY data/processed/ data/processed/" in dockerfile
    assert "COPY src/ src/" in dockerfile
    assert 'CMD ["uvicorn", "realtime_server:app"' in dockerfile
