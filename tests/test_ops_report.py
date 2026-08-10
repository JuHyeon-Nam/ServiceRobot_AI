import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


def test_ops_report_json_and_markdown_contract():
    from ops_report import build_ops_report, report_to_markdown

    snap = {
        "kpi": {"total": 2, "ok": 1, "warn": 1, "avg_health": 72, "maint_due": 1, "deteriorating": 1},
        "inference": {"mode": "live_booster", "last_latency_ms": 0.2},
    }
    risk = {
        "status": "watch",
        "score": 42.0,
        "bottleneck_floor": 0,
        "action_required": 1,
        "recommendation": "Keep monitoring.",
        "floor_risk": [{"floor": 0, "total": 2, "warn": 1, "avg_health": 72, "score": 42, "critical_assets": 0}],
        "top_assets": [{"id": "AGV-01", "floor": 0, "risk": 42, "label": "배터리 저하",
                        "pred": "E-RBT-B", "level": "경고", "health": 50, "trend_dir": "악화"}],
    }
    report = build_ops_report(
        100.1234,
        snap,
        risk,
        {"total": 3, "open_p1": 1, "overdue_open": 0, "open_by_priority": {"P1": 1}},
        {"status": "watch", "score": 2.1, "drifted_features": [], "watch_features": ["batt"]},
        {"availability": 0.98, "mtbf": 120.0, "mttr": 3.0, "episodes": 2},
        {"model_id": "robot-pdm-lightgbm-enhanced", "version": "robot_pdm_enhanced"},
    )

    assert report["schema"] == "fab.ops.report.v1"
    assert report["fleet"]["warn"] == 1
    assert report["risk"]["status"] == "watch"
    assert report["work_orders"]["open_p1"] == 1
    assert report["ai_ops"]["model_id"] == "robot-pdm-lightgbm-enhanced"

    md = report_to_markdown(report)
    assert "# FAB AGV Operations Report" in md
    assert "| AGV-01 | 0 | 42 | 배터리 저하 | 경고 | 50 | 악화 |" in md
    assert "Keep monitoring." in md
