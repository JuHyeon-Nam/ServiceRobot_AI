import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from rul_runtime import attach_rul_model_slot, rul_readiness_report


def test_rul_model_slot_contains_feature_vector():
    phm = {
        "stage": "watch",
        "severity": "주의",
        "risk_score": 42,
        "trend_slope": 0.75,
        "rul_estimate_min": 120,
    }
    sensors = {"vib": 3.4, "batt": 71.2, "temp": 39.8}

    out = attach_rul_model_slot(phm, sensors, health=76, warn=True)

    assert out["rul_model"]["schema"] == "fab.rul.calibration.v1"
    assert out["rul_model"]["mode"] == "heuristic_fallback"
    assert out["rul_model"]["ready_for_supervised"] is False
    assert out["rul_model"]["feature_vector"] == {
        "health": 76.0,
        "risk_score": 42.0,
        "trend_slope": 0.75,
        "vib": 3.4,
        "batt": 71.2,
        "temp": 39.8,
        "warn": 1,
        "severity_code": 1,
    }


def test_rul_readiness_report_requires_failure_time_labels():
    rows = [
        {
            "asset_id": "AGV-01",
            "event_ts": 10.0,
            "failure_ts": 20.0,
            "failure_code": "E-RBT-B",
            "censoring": False,
        },
        {
            "asset_id": "AGV-02",
            "event_ts": 10.0,
            "failure_ts": 0,
            "failure_code": "none",
            "censoring": True,
        },
    ]

    report = rul_readiness_report(rows)

    assert report["schema"] == "fab.rul.calibration.v1"
    assert report["ready_for_supervised"] is False
    assert report["failure_events"] == 1
    assert report["censored_windows"] == 1
    assert report["missing_required_fields"] == []
    assert report["checks"]["enough_failure_events"] is False
