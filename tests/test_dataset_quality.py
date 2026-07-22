import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from dataset_quality import RoboticsDataQualityMonitor, records_from_agv_snapshot
from drift_monitor import DataDriftMonitor


def test_quality_monitor_metrics():
    monitor = RoboticsDataQualityMonitor()
    records = [
        {
            "sample_id": "s1",
            "robot_id": "AGV-01",
            "timestamp": 1.0,
            "modality": "telemetry",
            "storage_uri": "memory://s1",
            "annotation": {"label": "정상"},
            "qa_status": "pass",
        },
        {
            "sample_id": "s2",
            "robot_id": "AGV-02",
            "timestamp": 2.0,
            "modality": "video",
            "storage_uri": "memory://s2",
            "annotation": {"label": "E-RBT-B"},
            "qa_status": "fail",
        },
        {
            "sample_id": "s3",
            "robot_id": "AGV-03",
            "timestamp": "bad",
            "modality": "unsupported",
            "storage_uri": "",
            "annotation": {},
            "qa_status": "maybe",
        },
    ]

    report = monitor.evaluate(records).as_dict()

    assert report["total"] == 3
    assert report["valid"] == 2
    assert report["invalid"] == 1
    assert report["annotation_coverage"] == 0.6667
    assert report["qa_pass_rate"] == 0.5
    assert report["ingest_success_rate"] == 0.6667
    assert report["rework_rate"] == 0.3333
    assert report["by_modality"] == {"telemetry": 1, "unsupported": 1, "video": 1}
    assert any(issue["field"] == "timestamp" for issue in report["issues"])


def test_records_from_agv_snapshot():
    agvs = [
        {
            "id": "AGV-01",
            "floor": 0,
            "status": "ok",
            "pred": "정상",
            "conf": 0.99,
            "health": 100,
            "trend_dir": "안정",
        },
        {
            "id": "AGV-02",
            "floor": 1,
            "status": "warn",
            "pred": "E-RBT-B",
            "conf": 0.55,
            "health": 42,
            "trend_dir": "악화",
        },
    ]

    records = records_from_agv_snapshot(agvs, 123.456)

    assert len(records) == 2
    assert records[0]["modality"] == "telemetry"
    assert records[0]["annotation"]["category"] == "normal"
    assert records[0]["qa_status"] == "pass"
    assert records[1]["annotation"]["category"] == "fault"
    assert records[1]["qa_status"] == "fail"


def test_data_drift_monitor_statuses():
    monitor = DataDriftMonitor()
    healthy = []
    for i in range(6):
        healthy.append({
            "id": f"AGV-{i + 1:02d}",
            "status": "warn" if i == 0 else "ok",  # reference fault rate is about 17%.
            "conf": 0.86,
            "health": 88,
            "sensors": {"vib": 2.2, "batt": 66, "temp": 42},
        })
    severe = [
        {"id": "AGV-01", "status": "warn", "conf": 0.98, "health": 25,
         "sensors": {"vib": 8.5, "batt": 12, "temp": 68}},
        {"id": "AGV-02", "status": "warn", "conf": 0.94, "health": 30,
         "sensors": {"vib": 7.9, "batt": 15, "temp": 66}},
    ]

    ok = monitor.evaluate(healthy)
    drift = monitor.evaluate(severe)

    assert ok["status"] == "ok"
    assert ok["window_size"] == 6
    assert drift["status"] == "drift"
    assert "fault_rate" in drift["drifted_features"]
    assert drift["score"] >= 3
    assert {"vib", "batt", "temp", "health", "conf"} <= drift["features"].keys()
