import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from demo_runtime import DemoRuntime
from telemetry_store import TelemetryStore
from work_order_store import WorkOrderStore


@pytest.fixture
def server(monkeypatch):
    import realtime_server as server
    monkeypatch.setattr(server, "DEMO", DemoRuntime())
    monkeypatch.setattr(server, "STORE", TelemetryStore())
    monkeypatch.setattr(server, "WORK_ORDERS", WorkOrderStore())
    monkeypatch.setattr(server, "EDGE_INPUTS", {})
    return server, TestClient(server.app)


def test_scenario_to_work_order_to_recovery(server):
    module, client = server
    asset_id = module.PLAN[0]["id"]
    for scenario, expected in (("normal", "normal"), ("watch", "watch"),
                               ("degrading", "predicted_fault"), ("battery", "current_fault")):
        assert client.post('/api/demo', json={"asset_id": asset_id, "scenario": scenario}).status_code == 200
        asset = next(a for a in client.get('/api/snapshot').json()["agvs"] if a["id"] == asset_id)
        assert asset["phm"]["stage"] == expected
        assert asset["source"] == asset["inference_mode"] == "demo_scenario"
        assert "model_diagnosis" in asset
    order = next(o for o in client.get('/api/work-orders').json()["orders"]
                 if o["agv"] == asset_id and o["fault"] == 'E-RBT-B')
    assert order["source"] == "demo_scenario"
    for status in ('acknowledged', 'in_progress', 'resolved'):
        assert client.post(f'/api/work-orders/{order["id"]}/status', params={"status": status}).status_code == 200
    for _ in range(3):
        rows = client.get('/api/work-orders').json()["orders"]
        assert len([o for o in rows if o["agv"] == asset_id and o["fault"] == 'E-RBT-B']) == 1
    client.post('/api/demo', json={"asset_id": asset_id, "scenario": "normal"})
    client.post('/api/demo', json={"asset_id": asset_id, "scenario": "battery"})
    rows = client.get('/api/work-orders').json()["orders"]
    assert len([o for o in rows if o["agv"] == asset_id and o["fault"] == 'E-RBT-B']) == 2
    client.post('/api/demo', json={"asset_id": asset_id, "scenario": "replay"})
    asset = next(a for a in client.get('/api/snapshot').json()["agvs"] if a["id"] == asset_id)
    assert asset["source"] == "replay_model"


def test_controls_validate_before_mutation(server):
    module, client = server
    assert client.post('/api/demo', json={"asset_id": "missing", "scenario": "battery", "paused": True}).status_code == 400
    assert not module.DEMO.paused
    for body in ({"speed": 5}, {"scenario": "not_real"}, {"unexpected": True}):
        assert client.post('/api/demo', json=body).status_code == 422
    assert client.post('/api/demo', json={"paused": True, "speed": 0.25}).status_code == 200
    assert client.get('/api/demo').json()["paused"]


def test_edge_input_takes_precedence(server):
    import time
    module, client = server
    asset_id = module.PLAN[0]["id"]
    client.post('/api/demo', json={"asset_id": asset_id, "scenario": "battery"})
    module.EDGE_INPUTS[asset_id] = {"ingested_at": time.time(), "payload": {
        "diagnosis": {"fault": "정상", "status": "ok", "confidence": 0.9},
        "health": {"index": 100}, "sensors": {"batt": 90, "vib": 1, "temp": 38}}}
    asset = next(a for a in client.get('/api/snapshot').json()["agvs"] if a["id"] == asset_id)
    assert asset["source"] == "edge_ingest"
    assert asset["pred"] == "정상"
    assert asset["scenario"] is None


def test_scenario_events_excluded_from_rul_training(server, tmp_path):
    from build_rul_dataset import read_events_csv, read_events_sqlite
    module, client = server
    asset_id = module.PLAN[0]["id"]
    client.post('/api/demo', json={"asset_id": asset_id, "scenario": "battery"})
    response = client.get('/api/history', params={"agv": asset_id, "fmt": "csv"})
    path = tmp_path / 'events.csv'
    path.write_text(response.text, encoding='utf-8')
    assert 'demo_scenario' in response.text
    assert read_events_csv(str(path)) == []
    db = tmp_path / 'events.db'
    store = TelemetryStore(str(db))
    assets = client.get('/api/snapshot').json()["agvs"]
    store.record(100, [a for a in assets if a["id"] == asset_id])
    assert store.events()[0]["source"] == "demo_scenario"
    assert read_events_sqlite(str(db)) == []


def test_scenario_provenance_survives_edge_roundtrip(server):
    from edge_gateway import payload_from_agv
    import time
    module, client = server
    asset_id = module.PLAN[0]["id"]
    client.post('/api/demo', json={"asset_id": asset_id, "scenario": "battery"})
    asset = next(a for a in client.get('/api/snapshot').json()["agvs"] if a["id"] == asset_id)
    payload = payload_from_agv(time.time(), asset)
    client.post('/api/demo', json={"asset_id": asset_id, "scenario": "replay"})
    assert client.post('/api/edge-ingest', json=payload).json()["accepted"]
    asset = next(a for a in client.get('/api/snapshot').json()["agvs"] if a["id"] == asset_id)
    assert asset["edge_input"]["active"]
    assert asset["source"] == "demo_scenario"
