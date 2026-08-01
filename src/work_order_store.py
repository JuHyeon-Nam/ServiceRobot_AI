"""
work_order_store.py — maintenance work-order layer for predicted faults.

Predictive maintenance becomes much more convincing when a warning turns into a
trackable maintenance action. This store keeps a lightweight CMMS-style queue on
SQLite without adding external services.
"""
from __future__ import annotations

import sqlite3
import threading


_DDL = """CREATE TABLE IF NOT EXISTS work_orders(
  id              TEXT PRIMARY KEY,
  created_ts      REAL,
  updated_ts      REAL,
  agv             TEXT,
  floor           INTEGER,
  fault           TEXT,
  level           TEXT,
  health          INTEGER,
  priority        TEXT,
  status          TEXT,
  title           TEXT,
  recommendation  TEXT,
  source          TEXT
)"""


OPEN_STATUSES = {"open", "acknowledged", "in_progress"}


def priority_for(level: str | None, health: int) -> str:
    if level == "위험" or health < 30:
        return "P1"
    if level == "경고" or health < 55:
        return "P2"
    return "P3"


def recommendation_for(fault: str, level: str | None, health: int) -> str:
    if fault == "E-RBT-B":
        return "Battery pack inspection: check cycle count, charge behavior, and replacement threshold."
    if fault == "E-RBT-E":
        return "Immediate safety inspection: verify E-stop chain, brake state, and blocked route condition."
    if fault == "E-RBT-N":
        return "Network inspection: check AP handoff, robot modem logs, and offline event duration."
    if fault == "E-RBT-S":
        return "Sensor inspection: recalibrate sensor module and compare vibration/heading traces."
    if fault in ("E-ENV-C", "E-ENV-O"):
        return "Route inspection: clear obstacle/collision zone and review local traffic density."
    if fault in ("E-INF-A", "E-INF-E"):
        return "Interface inspection: verify door/lift handshake logs and interlock state."
    if health < 55:
        return "Preventive inspection: low health index despite normal instantaneous classification."
    return "Monitor next windows and keep the order open until health stabilizes."


class WorkOrderStore:
    """Small work-order queue with idempotent sync from live AGV snapshots."""

    def __init__(self, path: str = ":memory:"):
        self.cx = sqlite3.connect(path, check_same_thread=False)
        self.cx.execute(_DDL)
        self.cx.execute("CREATE INDEX IF NOT EXISTS ix_work_orders_status ON work_orders(status)")
        self.cx.execute("CREATE INDEX IF NOT EXISTS ix_work_orders_agv ON work_orders(agv, status)")
        self.cx.commit()
        self.lock = threading.Lock()

    def sync_from_snapshot(self, ts: float, agvs: list[dict]) -> int:
        """Create/update open work orders for warning or low-health AGVs."""
        changed = 0
        for agv in agvs:
            fault = agv.get("pred", "정상")
            health = int(agv.get("health", 100))
            needs_order = agv.get("status") == "warn" or health < 55
            if not needs_order:
                continue
            level = agv.get("level") or "주의"
            order_id = f"WO-{agv['id']}-{fault}"
            priority = priority_for(level, health)
            title = f"{agv['id']} {agv.get('label', fault)} 점검"
            rec = recommendation_for(fault, level, health)
            with self.lock:
                row = self.cx.execute(
                    "SELECT status FROM work_orders WHERE id=?", (order_id,)
                ).fetchone()
                if row and row[0] in OPEN_STATUSES:
                    self.cx.execute(
                        "UPDATE work_orders SET updated_ts=?, floor=?, level=?, health=?, "
                        "priority=?, recommendation=? WHERE id=?",
                        (ts, agv.get("floor"), level, health, priority, rec, order_id),
                    )
                elif not row:
                    self.cx.execute(
                        "INSERT INTO work_orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (order_id, ts, ts, agv["id"], agv.get("floor"), fault, level, health,
                         priority, "open", title, rec, "live_booster"),
                    )
                else:
                    # A closed order is not reopened automatically; create a fresh revision.
                    order_id = f"{order_id}-{int(ts)}"
                    self.cx.execute(
                        "INSERT INTO work_orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (order_id, ts, ts, agv["id"], agv.get("floor"), fault, level, health,
                         priority, "open", title, rec, "live_booster"),
                    )
                self.cx.commit()
            changed += 1
        return changed

    def list(self, status: str | None = None, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM work_orders"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY CASE priority WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 ELSE 3 END, updated_ts DESC LIMIT ?"
        args = (*args, limit)
        with self.lock:
            cur = self.cx.execute(q, args)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def set_status(self, order_id: str, status: str, ts: float) -> dict | None:
        if status not in {"open", "acknowledged", "in_progress", "resolved", "closed"}:
            raise ValueError(f"unsupported_status:{status}")
        with self.lock:
            row = self.cx.execute("SELECT id FROM work_orders WHERE id=?", (order_id,)).fetchone()
            if not row:
                return None
            self.cx.execute(
                "UPDATE work_orders SET status=?, updated_ts=? WHERE id=?",
                (status, ts, order_id),
            )
            self.cx.commit()
        return self.get(order_id)

    def get(self, order_id: str) -> dict | None:
        with self.lock:
            cur = self.cx.execute("SELECT * FROM work_orders WHERE id=?", (order_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [c[0] for c in cur.description]
            return dict(zip(cols, row))

    def summary(self) -> dict:
        with self.lock:
            total = self.cx.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            by_status = dict(self.cx.execute(
                "SELECT status,COUNT(*) FROM work_orders GROUP BY status"
            ).fetchall())
            by_priority = dict(self.cx.execute(
                "SELECT priority,COUNT(*) FROM work_orders GROUP BY priority"
            ).fetchall())
            open_p1 = self.cx.execute(
                "SELECT COUNT(*) FROM work_orders WHERE status IN ('open','acknowledged','in_progress') "
                "AND priority='P1'"
            ).fetchone()[0]
        return {"total": total, "by_status": by_status, "by_priority": by_priority, "open_p1": open_p1}
