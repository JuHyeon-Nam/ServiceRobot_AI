"""
ops_report.py — compact operations report builder.

The realtime server already exposes many operational APIs. This module combines
the most important signals into one report that can be consumed by automation,
rendered as Markdown, or pasted into an incident log.
"""
from __future__ import annotations


def build_ops_report(
    ts: float,
    snapshot: dict,
    risk: dict,
    work_orders: dict,
    drift: dict,
    reliability: dict,
    model_card: dict,
) -> dict:
    kpi = snapshot.get("kpi", {})
    inference = snapshot.get("inference", {})
    top_assets = risk.get("top_assets", [])[:5]
    floor_risk = risk.get("floor_risk", [])

    return {
        "schema": "fab.ops.report.v1",
        "ts": round(float(ts), 3),
        "fleet": {
            "total": kpi.get("total", 0),
            "ok": kpi.get("ok", 0),
            "warn": kpi.get("warn", 0),
            "avg_health": kpi.get("avg_health", 0),
            "maint_due": kpi.get("maint_due", 0),
            "deteriorating": kpi.get("deteriorating", 0),
        },
        "risk": {
            "status": risk.get("status"),
            "score": risk.get("score"),
            "bottleneck_floor": risk.get("bottleneck_floor"),
            "action_required": risk.get("action_required", 0),
            "recommendation": risk.get("recommendation", ""),
        },
        "floor_risk": floor_risk,
        "top_assets": top_assets,
        "work_orders": {
            "total": work_orders.get("total", 0),
            "open_p1": work_orders.get("open_p1", 0),
            "overdue_open": work_orders.get("overdue_open", 0),
            "open_by_priority": work_orders.get("open_by_priority", {}),
        },
        "ai_ops": {
            "drift_status": drift.get("status"),
            "drift_score": drift.get("score"),
            "drifted_features": drift.get("drifted_features", []),
            "watch_features": drift.get("watch_features", []),
            "model_id": model_card.get("model_id"),
            "model_version": model_card.get("version"),
            "inference_mode": inference.get("mode"),
            "last_latency_ms": inference.get("last_latency_ms"),
        },
        "reliability": {
            "availability": reliability.get("availability"),
            "mtbf": reliability.get("mtbf"),
            "mttr": reliability.get("mttr"),
            "episodes": reliability.get("episodes", 0),
        },
    }


def report_to_markdown(report: dict) -> str:
    fleet = report["fleet"]
    risk = report["risk"]
    wo = report["work_orders"]
    ai = report["ai_ops"]
    rel = report["reliability"]

    lines = [
        "# FAB AGV Operations Report",
        "",
        f"- Timestamp: `{report['ts']}`",
        f"- Fleet: total `{fleet['total']}`, ok `{fleet['ok']}`, warn `{fleet['warn']}`",
        f"- Average health: `{fleet['avg_health']}`",
        f"- Risk: `{risk['status']}` / score `{risk['score']}` / bottleneck floor `{risk['bottleneck_floor']}`",
        f"- Action required assets: `{risk['action_required']}`",
        f"- Work orders: total `{wo['total']}`, open P1 `{wo['open_p1']}`, overdue `{wo['overdue_open']}`",
        f"- Drift: `{ai['drift_status']}` / score `{ai['drift_score']}`",
        f"- Reliability: availability `{rel['availability']}`, MTBF `{rel['mtbf']}`, MTTR `{rel['mttr']}`",
        "",
        "## Recommendation",
        "",
        risk["recommendation"] or "No recommendation.",
        "",
        "## Top Risk Assets",
        "",
        "| Asset | Floor | Risk | Fault | Level | Health | Trend |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for asset in report["top_assets"]:
        lines.append(
            f"| {asset['id']} | {asset['floor']} | {asset['risk']} | "
            f"{asset.get('label') or asset.get('pred')} | {asset.get('level')} | "
            f"{asset['health']} | {asset.get('trend_dir')} |"
        )

    lines += [
        "",
        "## Floor Risk",
        "",
        "| Floor | Total | Warn | Avg Health | Score | Critical Assets |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for floor in report["floor_risk"]:
        lines.append(
            f"| {floor['floor']} | {floor['total']} | {floor['warn']} | "
            f"{floor['avg_health']} | {floor['score']} | {floor['critical_assets']} |"
        )
    return "\n".join(lines) + "\n"


def build_shift_handover(report: dict, shift: str = "current") -> dict:
    fleet = report["fleet"]
    risk = report["risk"]
    wo = report["work_orders"]
    ai = report["ai_ops"]
    rel = report["reliability"]

    checklist = []
    if wo["overdue_open"] > 0:
        checklist.append({
            "priority": "P1",
            "category": "maintenance",
            "action": f"Escalate {wo['overdue_open']} overdue work orders.",
        })
    if wo["open_p1"] > 0:
        checklist.append({
            "priority": "P1",
            "category": "maintenance",
            "action": f"Acknowledge and dispatch {wo['open_p1']} open P1 work orders.",
        })
    if risk["action_required"] > 0:
        checklist.append({
            "priority": "P2",
            "category": "fleet",
            "action": f"Inspect top {min(risk['action_required'], 5)} risk assets before normal patrol.",
        })
    if ai["drift_status"] != "ok":
        checklist.append({
            "priority": "P2",
            "category": "ai_ops",
            "action": f"Review drift status `{ai['drift_status']}` and sample-label the current window.",
        })
    if not checklist:
        checklist.append({
            "priority": "P3",
            "category": "routine",
            "action": "Continue normal monitoring cadence.",
        })

    if risk["status"] == "critical" or wo["overdue_open"] > 0:
        status = "critical"
    elif risk["status"] == "watch" or wo["open_p1"] > 0 or fleet["warn"] > 0 or ai["drift_status"] != "ok":
        status = "watch"
    else:
        status = "ok"

    return {
        "schema": "fab.shift.handover.v1",
        "ts": report["ts"],
        "shift": shift,
        "status": status,
        "summary": [
            f"Fleet warn {fleet['warn']}/{fleet['total']}, avg health {fleet['avg_health']}.",
            f"Risk {risk['status']} score {risk['score']}, bottleneck floor {risk['bottleneck_floor']}.",
            f"Work orders total {wo['total']}, open P1 {wo['open_p1']}, overdue {wo['overdue_open']}.",
            f"Drift {ai['drift_status']} score {ai['drift_score']}, availability {rel['availability']}.",
        ],
        "checklist": checklist,
        "watch_assets": report["top_assets"][:3],
        "floor_focus": {
            "bottleneck_floor": risk["bottleneck_floor"],
            "floor_risk": report["floor_risk"],
        },
        "model": {
            "model_id": ai["model_id"],
            "inference_mode": ai["inference_mode"],
            "last_latency_ms": ai["last_latency_ms"],
        },
    }


def handover_to_markdown(handover: dict) -> str:
    lines = [
        "# FAB AGV Shift Handover",
        "",
        f"- Shift: `{handover['shift']}`",
        f"- Timestamp: `{handover['ts']}`",
        f"- Status: `{handover['status']}`",
        "",
        "## Summary",
        "",
    ]
    lines += [f"- {item}" for item in handover["summary"]]
    lines += [
        "",
        "## Checklist",
        "",
        "| Priority | Category | Action |",
        "|---|---|---|",
    ]
    for item in handover["checklist"]:
        lines.append(f"| {item['priority']} | {item['category']} | {item['action']} |")
    lines += [
        "",
        "## Watch Assets",
        "",
        "| Asset | Floor | Risk | Fault | Health |",
        "|---|---:|---:|---|---:|",
    ]
    for asset in handover["watch_assets"]:
        lines.append(
            f"| {asset['id']} | {asset['floor']} | {asset['risk']} | "
            f"{asset.get('label') or asset.get('pred')} | {asset['health']} |"
        )
    return "\n".join(lines) + "\n"
