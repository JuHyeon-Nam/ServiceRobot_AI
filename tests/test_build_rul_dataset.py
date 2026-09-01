import os
import sqlite3
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from build_rul_dataset import (
    build_rul_rows,
    dataset_metadata,
    main,
    normalize_event,
    read_events_csv,
    read_events_sqlite,
    read_failure_labels,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_build_rul_rows_joins_next_failure_and_censors_tail():
    events = [
        normalize_event({
            "agv": "AGV-01",
            "ts": "100",
            "health": "80",
            "risk_score": "20",
            "trend_slope": "0.1",
            "vib": "2.1",
            "batt": "70",
            "temp": "37",
            "pred": "정상",
        }),
        normalize_event({
            "agv": "AGV-01",
            "ts": "160",
            "health": "48",
            "risk_score": "74",
            "trend_slope": "1.2",
            "vib": "6.2",
            "batt": "24",
            "temp": "61",
            "pred": "E-RBT-B",
            "level": "경고",
        }),
        normalize_event({
            "agv": "AGV-02",
            "ts": "200",
            "health": "99",
            "vib": "1.9",
            "batt": "88",
            "temp": "35",
            "pred": "정상",
        }),
    ]
    labels = [{"asset_id": "AGV-01", "failure_ts": 220.0, "failure_code": "E-RBT-B", "censoring": False}]

    rows = build_rul_rows(events, labels)

    assert len(rows) == 3
    assert rows[0]["asset_id"] == "AGV-01"
    assert rows[0]["minutes_to_failure"] == 2.0
    assert rows[1]["minutes_to_failure"] == 1.0
    assert rows[2]["asset_id"] == "AGV-02"
    assert rows[2]["event_observed"] == 0
    assert rows[2]["censoring"] is True
    assert rows[1]["risk_score"] == 74.0 and rows[1]["trend_slope"] == 1.2

    meta = dataset_metadata(rows)
    assert meta["schema"] == "fab.rul.calibration.v1"
    assert meta["failure_rows"] == 2
    assert meta["censored_rows"] == 1
    assert "risk_score" in meta["feature_fields"]


def test_read_events_csv_accepts_telemetry_export_columns(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "ts,agv,pred,level,health,vib,batt,temp,risk_score,trend_slope\n"
        "100,AGV-01,E-RBT-S,위험,22,9.1,40,64,97,1.5\n",
        encoding="utf-8",
    )

    rows = read_events_csv(str(path))

    assert rows == [{
        "asset_id": "AGV-01",
        "event_ts": 100.0,
        "features": {
            "health": 22.0,
            "risk_score": 97.0,
            "trend_slope": 1.5,
            "vib": 9.1,
            "batt": 40.0,
            "temp": 64.0,
            "warn": 1,
            "severity_code": 3,
        },
    }]


def test_read_events_sqlite_accepts_older_telemetry_schema(tmp_path):
    db = tmp_path / "telemetry.sqlite"
    with sqlite3.connect(db) as cx:
        cx.execute(
            "CREATE TABLE events(ts REAL, agv TEXT, floor INTEGER, pred TEXT, conf REAL, "
            "level TEXT, health INTEGER, vib REAL, batt REAL, temp REAL)"
        )
        cx.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
            (100.0, "AGV-01", 0, "E-RBT-B", 0.92, "위험", 25, 7.2, 18.0, 62.0),
        )

    rows = read_events_sqlite(str(db))

    assert len(rows) == 1
    assert rows[0]["asset_id"] == "AGV-01"
    assert rows[0]["features"]["trend_slope"] == 0.0
    assert rows[0]["features"]["risk_score"] > 0


def test_build_rul_dataset_cli_smoke_fixture(tmp_path):
    out_csv = tmp_path / "rul_training.csv"
    meta_json = tmp_path / "rul_training.metadata.json"

    rc = main([
        "--events-csv", os.path.join(FIXTURES, "rul_events.csv"),
        "--labels-csv", os.path.join(FIXTURES, "rul_failure_labels.csv"),
        "--out-csv", str(out_csv),
        "--meta-json", str(meta_json),
    ])

    assert rc == 0
    assert out_csv.exists() and meta_json.exists()
    events = read_events_csv(os.path.join(FIXTURES, "rul_events.csv"))
    labels = read_failure_labels(os.path.join(FIXTURES, "rul_failure_labels.csv"))
    rows = build_rul_rows(events, labels)
    assert len(rows) == 13
    assert sum(row["event_observed"] == 1 for row in rows) == 12
    assert sum(row["event_observed"] == 0 for row in rows) == 1
