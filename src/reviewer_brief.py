"""
reviewer_brief.py — portfolio reviewer walkthrough metadata.

This is intentionally small and static: it gives recruiters, professors, and
interviewers a fast path through the demo without reading the whole README.
"""
from __future__ import annotations


def build_reviewer_brief() -> dict:
    return {
        "project": "ServiceRobot_AI",
        "one_liner": (
            "A real-time predictive-maintenance AI control center for service "
            "robots and FAB AGV fleets, with a tablet-operable 3D digital twin."
        ),
        "review_time_minutes": 3,
        "start_here": [
            {
                "step": 1,
                "title": "Open the 3D twin",
                "path": "/twin",
                "look_for": "3-floor FAB layout, moving AGVs, severity colors, warning beams, camera focus.",
            },
            {
                "step": 2,
                "title": "Click one AGV",
                "path": "/twin",
                "look_for": "Sensor traces, health index, code-based inspection hints, source labels, and event history.",
            },
            {
                "step": 3,
                "title": "Complete a maintenance workflow and inspect model evidence",
                "path": "/twin",
                "look_for": "Maintenance tab: acknowledge, inspect, resolve. Data/AI tab: validation metrics and simulation boundaries.",
            },
            {
                "step": 4,
                "title": "Check reproducibility",
                "path": "docker compose up --build",
                "look_for": "One-command deploy with durable SQLite telemetry volume.",
            },
        ],
        "proof_points": [
            {
                "claim": "Model generalization is measured honestly",
                "evidence": "Official validation split reports unseen-robot accuracy separately from random split.",
            },
            {
                "claim": "Inference is operationally lightweight",
                "evidence": "LightGBM native artifact, CPU-only serving, 249-feature contract, no GPU dependency.",
            },
            {
                "claim": "Dashboard is a real control surface",
                "evidence": "WebSocket state stream, 3D twin, severity triage, health index, alert focus.",
            },
            {
                "claim": "AI diagnosis is tied to maintenance execution",
                "evidence": "Predicted faults and low-health assets become P1/P2/P3 work orders with status tracking.",
            },
            {
                "claim": "Data engineering is represented",
                "evidence": "MQTT-style edge messages, physical sensor adapter, SQLite event store, rollups, history query, CSV export, retention policy.",
            },
            {
                "claim": "MLOps and governance are represented",
                "evidence": "Prometheus metrics, data QA, data drift, reliability metrics, model card API.",
            },
            {
                "claim": "Deployment is reproducible",
                "evidence": "Dockerfile, docker-compose.yml, pytest contracts, GitHub Actions CI.",
            },
        ],
        "role_mapping": [
            {
                "role": "Data / AI Engineer",
                "keywords": ["ETL", "feature engineering", "model serving", "MLOps", "data quality", "edge telemetry"],
                "evidence": ["/predict", "/api/edge-contract", "/api/edge-events", "/api/data-quality", "/api/drift", "/api/model-card"],
            },
            {
                "role": "Robotics / Smart Factory Engineer",
                "keywords": ["AGV", "AMHS", "digital twin", "predictive maintenance", "fleet monitoring", "MQTT"],
                "evidence": ["/twin", "/ws", "/api/snapshot", "/api/edge-events", "/api/work-orders", "fab_layout.py"],
            },
            {
                "role": "Backend / Platform Engineer",
                "keywords": ["FastAPI", "WebSocket", "Docker", "healthcheck", "observability"],
                "evidence": ["realtime_server.py", "docker-compose.yml", "/metrics"],
            },
            {
                "role": "Semiconductor Equipment / Operations",
                "keywords": ["FAB", "equipment health", "MTBF", "MTTR", "availability", "work order"],
                "evidence": ["/api/work-orders", "/api/reliability", "/api/history", "/api/trend", "/twin"],
            },
        ],
        "demo_script": [
            "0:00-0:20 Run docker compose or uvicorn and open /twin.",
            "0:20-0:55 Show moving AGVs, floors, OHT rails, lift, and severity colors.",
            "0:55-1:30 Select AGV-01 and apply normal, watch, degrading, and battery scenarios; identify these as scripted states.",
            "1:30-2:05 Use the Maintenance tab to acknowledge, start inspection, and resolve a work order.",
            "2:05-2:35 Use the Data/AI tab to explain official validation, macro-F1, and heuristic PHM.",
            "2:35-3:00 Return to model replay and export the shift handover report.",
        ],
        "current_status": {
            "portfolio_demo": "Integrated simulation, diagnosis, and maintenance workflow implemented; see docs/PROJECT_STATUS.md for verification.",
            "production_like_robotics_data_platform": "Prototype; physical integration, field RUL calibration, and deployment hardening remain unvalidated.",
            "next_best_work": "Collect independent failure-time labels and validate a physical sensor-to-twin path.",
        },
    }
