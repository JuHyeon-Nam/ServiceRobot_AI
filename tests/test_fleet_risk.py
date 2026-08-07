import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


def _agv(**overrides):
    row = {
        "id": "AGV-01",
        "floor": 0,
        "status": "ok",
        "pred": "정상",
        "label": "정상",
        "level": None,
        "health": 100,
        "trend_dir": "안정",
    }
    row.update(overrides)
    return row


def test_asset_risk_responds_to_fault_health_and_trend():
    from fleet_risk import asset_risk

    normal = asset_risk(_agv())
    critical = asset_risk(_agv(
        status="warn",
        pred="E-RBT-E",
        label="긴급정지",
        level="위험",
        health=24,
        trend_dir="악화",
    ))

    assert normal["risk"] == 0
    assert critical["risk"] > 70
    assert {"긴급정지", "low_health", "deteriorating"} <= set(critical["reasons"])


def test_fleet_risk_identifies_bottleneck_floor_and_work_order_pressure():
    from fleet_risk import fleet_risk

    agvs = [
        _agv(id="AGV-01", floor=0, status="warn", pred="E-RBT-B", label="배터리 저하",
             level="위험", health=30, trend_dir="악화"),
        _agv(id="AGV-02", floor=0, status="warn", pred="E-ENV-O", label="경로 장애물",
             level="경고", health=50, trend_dir="안정"),
        _agv(id="AGV-03", floor=1, health=95),
    ]

    out = fleet_risk(agvs, {"open_p1": 2, "overdue_open": 1})

    assert out["status"] in {"watch", "critical"}
    assert out["score"] > max(f["score"] for f in out["floor_risk"])  # work-order pressure is included
    assert out["bottleneck_floor"] == 0
    assert out["top_assets"][0]["id"] == "AGV-01"
    assert out["work_orders"] == {"open_p1": 2, "overdue_open": 1}


def test_fleet_risk_status_respects_action_pressure():
    from fleet_risk import fleet_risk

    watch = fleet_risk([_agv(id="AGV-01", status="warn", pred="E-ENV-O", label="경로 장애물",
                             level="주의", health=80)], {"open_p1": 1, "overdue_open": 0})
    critical = fleet_risk([_agv()], {"open_p1": 0, "overdue_open": 1})

    assert watch["status"] == "watch"
    assert critical["status"] == "critical"
