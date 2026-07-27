# Project Status — ServiceRobot_AI

## Ultimate Target

The final portfolio artifact is a **tablet-operable real-time 3D digital twin
and predictive-maintenance control center** for service robots / FAB AGV fleets.

It should show four capabilities in one coherent demo:

1. **AI PdM model**: diagnose normal + 8 fault states from robot sensor windows.
2. **Real-time operations UI**: stream AGV state into a 2D/3D control center.
3. **Data/AI operations layer**: persist telemetry, query history, monitor drift,
   expose reliability metrics, and document model governance.
4. **Portable deployment**: run the whole demo on another machine with one
   command and no original 4.2GB dataset.

## Current Completion

| Area | Status | Progress |
|---|---|---:|
| Core PdM model and honest validation | Done | 100% |
| Explainability and feature contract | Done | 100% |
| 3D digital twin demo | Done | 100% |
| Telemetry storage / history / rollup | Done | 100% |
| Reliability, metrics, drift, model card | Done | 100% |
| Portable deployment | Docker + compose done; local Docker unavailable here for manual run | 85% |
| Physical/edge realism | MQTT / external TSDB still pending | 45% |
| Portfolio packaging | README/docs strong; demo video still pending | 75% |

Overall: **about 85% complete as a portfolio demo**, and **about 70% complete as
a production-like robotics data platform**.

## What Is Already Demo-Ready

- `/twin`: 3D FAB/AGV digital twin with touch navigation and warning highlights.
- `/ws`: live fleet state stream.
- `/api/snapshot`: current AGV state and KPI contract.
- `/api/history`, `/api/stats`, `/api/trend`: telemetry persistence and analysis.
- `/api/reliability`: MTBF, MTTR, availability.
- `/api/data-quality`: robotics dataset QA/governance metrics.
- `/api/drift`: live data drift monitoring.
- `/api/model-card`: model artifact hash, feature contract, metrics, limitations.
- `/metrics`: Prometheus-compatible monitoring.
- `docker compose up --build`: one-command deployment entrypoint.

## Remaining Work

| Priority | Work | Why it matters |
|---|---|---|
| P1 | Demo video, 2-3 minutes | Makes the project instantly reviewable in portfolio/resume contexts. |
| P1 | README job-keyword polish | Maps features to data/AI, MLOps, robotics, and semiconductor operations roles. |
| P2 | MQTT simulator split | Makes ingestion look like a realistic edge-to-cloud architecture. |
| P2 | External time-series DB design or optional profile | Shows scaling path beyond SQLite. |
| P3 | True live `booster.predict` in the stream loop | Removes the replay-prediction caveat, useful if physical/edge data is added. |

## Recommended Next Three Daily Commits

1. `docs(portfolio): add reviewer walkthrough and role mapping`
2. `feat(edge): add MQTT-style simulator interface`
3. `docs(demo): add video script and capture checklist`
