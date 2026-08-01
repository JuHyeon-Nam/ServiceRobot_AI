import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


def _agv(**overrides):
    row = {
        "id": "AGV-01",
        "floor": 1,
        "status": "warn",
        "pred": "E-RBT-B",
        "label": "배터리 저하",
        "level": "위험",
        "health": 24,
    }
    row.update(overrides)
    return row


def test_priority_for_triage_contract():
    from work_order_store import priority_for

    assert priority_for("위험", 80) == "P1"
    assert priority_for(None, 20) == "P1"
    assert priority_for("경고", 90) == "P2"
    assert priority_for("주의", 54) == "P2"
    assert priority_for("주의", 90) == "P3"


def test_sync_creates_and_updates_open_work_order():
    from work_order_store import WorkOrderStore

    store = WorkOrderStore(":memory:")
    assert store.sync_from_snapshot(100.0, [_agv()]) == 1
    orders = store.list()

    assert len(orders) == 1
    assert orders[0]["id"] == "WO-AGV-01-E-RBT-B"
    assert orders[0]["priority"] == "P1"
    assert orders[0]["status"] == "open"
    assert "Battery pack" in orders[0]["recommendation"]

    store.sync_from_snapshot(110.0, [_agv(level="경고", health=42)])
    updated = store.get("WO-AGV-01-E-RBT-B")
    assert updated["updated_ts"] == 110.0
    assert updated["priority"] == "P2"
    assert store.summary()["total"] == 1


def test_sync_creates_low_health_order_without_current_fault():
    from work_order_store import WorkOrderStore

    store = WorkOrderStore(":memory:")
    changed = store.sync_from_snapshot(
        200.0,
        [_agv(status="ok", pred="정상", label="정상", level=None, health=50)],
    )

    assert changed == 1
    order = store.list()[0]
    assert order["priority"] == "P2"
    assert "Preventive inspection" in order["recommendation"]


def test_status_transitions_and_closed_revision():
    from work_order_store import WorkOrderStore

    store = WorkOrderStore(":memory:")
    store.sync_from_snapshot(100.0, [_agv()])
    row = store.set_status("WO-AGV-01-E-RBT-B", "in_progress", 120.0)
    assert row["status"] == "in_progress"

    assert store.set_status("missing", "closed", 130.0) is None
    store.set_status("WO-AGV-01-E-RBT-B", "closed", 140.0)
    store.sync_from_snapshot(150.0, [_agv()])

    assert store.summary()["total"] == 2
    assert any(o["id"] == "WO-AGV-01-E-RBT-B-150" for o in store.list(limit=10))


def test_invalid_status_rejected():
    from work_order_store import WorkOrderStore

    store = WorkOrderStore(":memory:")
    store.sync_from_snapshot(100.0, [_agv()])
    try:
        store.set_status("WO-AGV-01-E-RBT-B", "waiting_vendor", 120.0)
    except ValueError as e:
        assert "unsupported_status" in str(e)
    else:
        raise AssertionError("invalid status should raise ValueError")
