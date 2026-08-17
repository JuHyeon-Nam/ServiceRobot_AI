import json
import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


def _row(**overrides):
    r = {
        "ts": 100.123,
        "agv": "AGV-01",
        "floor": 2,
        "pred": "E-RBT-B",
        "conf": 0.91,
        "level": "위험",
        "health": 24,
        "vib": 4.1,
        "batt": 19.5,
        "temp": 51.2,
    }
    r.update(overrides)
    return r


def test_tsdb_contract_lists_supported_formats():
    from tsdb_export import tsdb_contract

    c = tsdb_contract()

    assert c["schema"] == "fab.telemetry.tsdb_export.v1"
    assert c["supported_formats"] == ["json", "influx", "timescale"]
    assert c["influx"]["measurement"] == "robot_pdm_events"
    assert c["timescale"]["hypertable"] is True


def test_influx_line_protocol_export():
    from tsdb_export import to_influx_lines

    text = to_influx_lines([_row()])

    assert text.startswith("robot_pdm_events,site=demo-fab,line=service-robot-ai,agv=AGV-01")
    assert "pred=E-RBT-B" in text
    assert "level=위험" in text
    assert "conf=0.91,health=24i,vib=4.1,batt=19.5,temp=51.2" in text
    assert text.rstrip().endswith("100123000000")


def test_timescale_sql_export():
    from tsdb_export import to_timescale_sql

    sql = to_timescale_sql([_row(agv="AGV'01")])

    assert "CREATE TABLE IF NOT EXISTS robot_pdm_events" in sql
    assert "create_hypertable('robot_pdm_events', 'ts'" in sql
    assert "INSERT INTO robot_pdm_events" in sql
    assert "'AGV''01'" in sql


def test_json_export_payload():
    from tsdb_export import export_payload

    body, media = export_payload([_row()], fmt="json")
    parsed = json.loads(body)

    assert media.startswith("application/json")
    assert parsed["schema"] == "fab.telemetry.tsdb_export.v1"
    assert parsed["events"][0]["agv"] == "AGV-01"
