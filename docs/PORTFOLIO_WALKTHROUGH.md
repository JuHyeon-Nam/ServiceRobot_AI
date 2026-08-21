# Portfolio Walkthrough

## 3-Minute Reviewer Path

1. Open `/twin`.
   Show the 3-floor FAB layout, moving AGVs, severity colors, warning beams, OHT rails, and lift.

2. Click one warning AGV.
   Point out live vibration/battery/temperature, AI cause text, health index, maintenance advice, and event history.

3. Open the operations APIs.
   Use `/api/edge-events`, `/api/work-orders`, `/api/reliability`, `/api/trend`, and `/api/history` to show edge telemetry topics, maintenance actions, MTBF/MTTR, fleet trends, and stored telemetry.

4. Open the AI governance APIs.
   Use `/api/drift`, `/api/data-quality`, and `/api/model-card` to show drift monitoring, data QA, artifact hash, and feature contract.

5. Close with reproducibility.
   Show `docker compose up --build` and the durable SQLite telemetry volume.

## Role Mapping

| Target role | What this project proves | Evidence |
|---|---|---|
| Data / AI Engineer | Feature engineering, honest validation, serving, MLOps, data quality, edge telemetry | `/predict`, `/api/edge-contract`, `/api/edge-events`, `/api/data-quality`, `/api/drift`, `/api/model-card` |
| Robotics / Smart Factory Engineer | AGV/FAB digital twin, fleet monitoring, fault triage, maintenance action loop | `/twin`, `/ws`, `/api/snapshot`, `/api/edge-events`, `/api/work-orders`, `src/fab_layout.py` |
| Backend / Platform Engineer | FastAPI, WebSocket, Docker, healthcheck, observability | `src/realtime_server.py`, `docker-compose.yml`, `/metrics` |
| Semiconductor Equipment / Operations | Health index, MTBF/MTTR, availability, maintenance priority and work-order status | `/api/work-orders`, `/api/reliability`, `/api/history`, `/api/trend`, `/twin` |

## Interview Talking Points

- I intentionally report official validation and random split separately because robot-level generalization matters more than a single high score.
- I removed absolute `x/y` coordinates from model inputs to reduce site-coordinate memorization and improve portability to new layouts.
- The dashboard is backed by a real WebSocket state stream and telemetry store, not only static screenshots.
- The edge telemetry contract exposes MQTT-compatible topics and validated payloads before a real broker is introduced.
- Predicted faults are converted into P1/P2/P3 work orders, so the demo connects AI diagnosis to maintenance execution.
- Operational AI reliability is covered with drift monitoring, data QA, Prometheus metrics, and a machine-readable model card.
- Docker Compose turns the demo into a reproducible artifact with durable telemetry storage.

## Demo Script

| Time | Narration |
|---|---|
| 0:00-0:20 | "This is a real-time PdM control center for service robots / FAB AGVs." |
| 0:20-0:55 | "The 3D twin streams AGV positions and AI status in real time." |
| 0:55-1:30 | "When I select an AGV, I can inspect sensors, health index, diagnosis reason, and history." |
| 1:30-2:05 | "The backend emits MQTT-style edge messages, turns warnings into maintenance work orders, and exposes trend/reliability analytics." |
| 2:05-2:35 | "For AI operations, I monitor data quality, live drift, and model artifact lineage." |
| 2:35-3:00 | "The whole demo runs with Docker Compose, and the scale-up path is MQTT broker ingest plus a time-series DB." |

## Current Completion

Portfolio demo: **98% complete**.

The core experience is demo-ready. The remaining highest-impact work is recording
a polished 2-3 minute demo video and adding a small physical-sensor adapter
example on top of the MQTT ingest path.
