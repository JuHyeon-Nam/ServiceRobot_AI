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
                "look_for": "Live vibration, battery, temperature, health index, AI cause text, event history.",
            },
            {
                "step": 3,
                "title": "Inspect data/AI operations endpoints",
                "path": "/api/reliability, /api/drift, /api/data-quality, /api/model-card",
                "look_for": "MTBF/MTTR, data drift, QA metrics, model artifact lineage and feature contract.",
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
                "claim": "Data engineering is represented",
                "evidence": "SQLite event store, rollups, history query, CSV export, retention policy.",
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
                "keywords": ["ETL", "feature engineering", "model serving", "MLOps", "data quality"],
                "evidence": ["/predict", "/api/data-quality", "/api/drift", "/api/model-card"],
            },
            {
                "role": "Robotics / Smart Factory Engineer",
                "keywords": ["AGV", "AMHS", "digital twin", "predictive maintenance", "fleet monitoring"],
                "evidence": ["/twin", "/ws", "/api/snapshot", "fab_layout.py"],
            },
            {
                "role": "Backend / Platform Engineer",
                "keywords": ["FastAPI", "WebSocket", "Docker", "healthcheck", "observability"],
                "evidence": ["realtime_server.py", "docker-compose.yml", "/metrics"],
            },
            {
                "role": "Semiconductor Equipment / Operations",
                "keywords": ["FAB", "equipment health", "MTBF", "MTTR", "availability", "triage"],
                "evidence": ["/api/reliability", "/api/history", "/api/trend", "/twin"],
            },
        ],
        "demo_script": [
            "0:00-0:20 Run docker compose or uvicorn and open /twin.",
            "0:20-0:55 Show moving AGVs, floors, OHT rails, lift, and severity colors.",
            "0:55-1:30 Click a warning AGV and explain health index, sensors, cause, and event history.",
            "1:30-2:05 Open /api/reliability and /api/trend to show operational analytics.",
            "2:05-2:35 Open /api/drift and /api/data-quality to show AI/data monitoring.",
            "2:35-3:00 Open /api/model-card and close with artifact lineage and remaining MQTT/TSDB path.",
        ],
        "current_status": {
            "portfolio_demo": "88%",
            "production_like_robotics_data_platform": "72%",
            "next_best_work": "Record a 2-3 minute demo video and add MQTT-style ingestion split.",
        },
    }
