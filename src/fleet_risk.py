"""
fleet_risk.py — operations-level fleet risk scoring.

This layer summarizes raw AGV diagnosis into a compact dispatch signal:
overall risk, floor bottleneck, top risky assets, and recommended action.
"""
from __future__ import annotations


LEVEL_WEIGHT = {None: 0.0, "주의": 12.0, "경고": 22.0, "위험": 35.0}
TREND_WEIGHT = {"안정": 0.0, "개선": -4.0, "악화": 10.0}


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def asset_risk(agv: dict) -> dict:
    health = int(agv.get("health", 100))
    level = agv.get("level")
    trend = agv.get("trend_dir", "안정")
    warn = agv.get("status") == "warn"
    health_penalty = max(0, 70 - health) * 0.8
    score = LEVEL_WEIGHT.get(level, 8.0 if warn else 0.0) + health_penalty + TREND_WEIGHT.get(trend, 0.0)
    if agv.get("pred") == "E-RBT-E":
        score += 10.0
    score = round(_clamp(score), 1)
    reasons = []
    if warn:
        reasons.append(agv.get("label") or agv.get("pred"))
    if health < 55:
        reasons.append("low_health")
    if trend == "악화":
        reasons.append("deteriorating")
    return {
        "id": agv.get("id"),
        "floor": agv.get("floor"),
        "risk": score,
        "pred": agv.get("pred"),
        "label": agv.get("label"),
        "level": level,
        "health": health,
        "trend_dir": trend,
        "reasons": reasons,
    }


def fleet_risk(agvs: list[dict], work_orders: dict | None = None) -> dict:
    assets = [asset_risk(a) for a in agvs]
    floors: dict[int, list[dict]] = {}
    for asset in assets:
        floors.setdefault(asset["floor"], []).append(asset)

    floor_risk = []
    for floor, rows in sorted(floors.items()):
        avg_health = round(sum(r["health"] for r in rows) / max(len(rows), 1), 1)
        score = round(sum(r["risk"] for r in rows) / max(len(rows), 1), 1)
        floor_risk.append({
            "floor": floor,
            "total": len(rows),
            "warn": sum(bool(r["level"]) for r in rows),
            "avg_health": avg_health,
            "score": score,
            "critical_assets": sum(r["risk"] >= 60 for r in rows),
        })

    top_assets = sorted(assets, key=lambda r: r["risk"], reverse=True)[:5]
    base_score = max([f["score"] for f in floor_risk], default=0.0)
    work_orders = work_orders or {}
    score = base_score
    score += min(int(work_orders.get("open_p1", 0)) * 3, 15)
    score += min(int(work_orders.get("overdue_open", 0)) * 8, 24)
    score = round(_clamp(score), 1)

    action_required = sum(1 for a in assets if a["risk"] >= 35)
    open_p1 = int(work_orders.get("open_p1", 0))
    overdue_open = int(work_orders.get("overdue_open", 0))

    if score >= 65 or overdue_open > 0:
        status = "critical"
        recommendation = "Dispatch maintenance now and protect the highest-risk floor."
    elif score >= 35 or action_required > 0 or open_p1 > 0:
        status = "watch"
        recommendation = "Keep monitoring and acknowledge P1/P2 work orders."
    else:
        status = "ok"
        recommendation = "Continue normal operation."

    bottleneck = max(floor_risk, key=lambda f: f["score"], default=None)
    return {
        "status": status,
        "score": score,
        "bottleneck_floor": bottleneck["floor"] if bottleneck else None,
        "floor_risk": floor_risk,
        "top_assets": top_assets,
        "action_required": action_required,
        "work_orders": {
            "open_p1": open_p1,
            "overdue_open": overdue_open,
        },
        "recommendation": recommendation,
    }
