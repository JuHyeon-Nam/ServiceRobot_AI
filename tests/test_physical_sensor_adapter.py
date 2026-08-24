import io
import json
import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


def test_parse_json_sensor_line_and_payload_contract():
    from edge_gateway import validate_payload
    from physical_sensor_adapter import SensorAdapterConfig, parse_sensor_line, payload_from_sensor_record

    cfg = SensorAdapterConfig(dry_run=True)
    row = parse_sensor_line('{"asset_id":"AGV-01","floor":2,"vib":6.2,"batt":44,"temp":61}', cfg)
    payload = payload_from_sensor_record(row, cfg, ts=123.456)

    assert payload["schema"] == "fab.edge.telemetry.v1"
    assert payload["asset_id"] == "AGV-01"
    assert payload["floor"] == 2
    assert payload["diagnosis"]["status"] == "warn"
    assert payload["diagnosis"]["fault"] == "E-RBT-S"
    assert payload["health"]["index"] < 100
    assert validate_payload(payload) == []


def test_parse_csv_sensor_line_uses_default_position_fields():
    from physical_sensor_adapter import SensorAdapterConfig, parse_sensor_line, payload_from_sensor_record

    cfg = SensorAdapterConfig(default_asset="AGV-X", default_floor=1)
    row = parse_sensor_line("AGV-02,1,2.1,88,38,120,80,270", cfg)
    payload = payload_from_sensor_record(row, cfg, ts=10.0)

    assert payload["asset_id"] == "AGV-02"
    assert payload["position"] == {"x": 120.0, "y": 80.0, "heading_deg": 270.0}
    assert payload["diagnosis"]["fault"] == "정상"
    assert payload["health"]["advice"] == "정상 운전"


def test_sensor_adapter_dry_run_does_not_post():
    from physical_sensor_adapter import SensorAdapterConfig, parse_sensor_line, payload_from_sensor_record, post_payload

    cfg = SensorAdapterConfig(dry_run=True)
    row = parse_sensor_line('{"asset_id":"AGV-03","floor":0,"vib":1.8,"batt":91,"temp":37}', cfg)
    result = post_payload(payload_from_sensor_record(row, cfg, ts=1.0), cfg)

    assert result["accepted"] is True
    assert result["mode"] == "dry_run"
    assert result["topic"].endswith("/AGV-03/telemetry")


def test_sensor_adapter_posts_to_edge_ingest():
    from physical_sensor_adapter import SensorAdapterConfig, parse_sensor_line, payload_from_sensor_record, post_payload

    calls = []

    def fake_post(url, params, json, timeout):
        calls.append((url, params, json, timeout))
        return FakeResponse({"accepted": True, "asset_id": json["asset_id"]})

    cfg = SensorAdapterConfig(ingest_url="http://api.local/api/edge-ingest")
    row = parse_sensor_line('{"asset_id":"AGV-04","floor":1,"vib":7.4,"batt":31,"temp":52}', cfg)
    result = post_payload(payload_from_sensor_record(row, cfg, ts=2.0), cfg, http_post=fake_post)

    assert result["accepted"] is True
    assert result["mode"] == "http_ingest"
    assert calls[0][0] == "http://api.local/api/edge-ingest"
    assert calls[0][1]["topic"] == "factory/demo-fab/floor/1/agv/AGV-04/telemetry"


def test_run_stream_accepts_multiple_physical_lines(capsys):
    from physical_sensor_adapter import SensorAdapterConfig, run_stream

    cfg = SensorAdapterConfig(dry_run=True)
    stream = io.StringIO(
        '{"asset_id":"AGV-05","floor":0,"vib":2.0,"batt":90,"temp":36}\n'
        "AGV-06,1,6.5,40,48,10,20,90\n"
    )
    results = run_stream(stream, cfg)
    out = capsys.readouterr().out

    assert len(results) == 2
    assert all(r["accepted"] for r in results)
    assert "AGV-05" in out and "AGV-06" in out


def test_edge_contract_exposes_physical_sensor_adapter():
    from edge_gateway import edge_contract

    assert edge_contract()["physical_sensor_adapter"] == "src/physical_sensor_adapter.py"
