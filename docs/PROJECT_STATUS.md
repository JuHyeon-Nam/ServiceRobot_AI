# Project Status — ServiceRobot_AI

## Ultimate Target

The final portfolio artifact is a **tablet-operable real-time 3D digital twin
and predictive-maintenance control center** for service robots / FAB AGV fleets.

It should show four capabilities in one coherent demo:

1. **AI PdM model**: diagnose normal + 8 fault states from robot sensor windows.
2. **Real-time operations UI**: stream AGV state into a 2D/3D control center.
3. **Data/AI operations layer**: persist telemetry, query history, monitor drift,
   expose reliability metrics, turn alerts into maintenance work orders, and
   document model governance.
4. **Edge-to-cloud realism**: expose an MQTT-compatible telemetry contract that
   can be replaced by a real broker and time-series DB.
5. **Portable deployment**: run the whole demo on another machine with one
   command and no original 4.2GB dataset.

## Current Completion

| Area | Status | Progress |
|---|---|---:|
| Core PdM model and honest validation | Done | 100% |
| Explainability and feature contract | Done | 100% |
| 3D digital twin demo | Done | 100% |
| Live inference in twin stream | Done | 100% |
| MQTT-style edge telemetry contract | Done | 100% |
| Optional MQTT broker bridge | Done; requires external broker and paho-mqtt for real publish | 85% |
| Telemetry storage / history / rollup | Done | 100% |
| External TSDB export contract | Done; Influx line protocol and Timescale SQL export | 85% |
| Predictive maintenance work orders | Done | 100% |
| PHM forecast contract | Done; heuristic RUL can be replaced by calibrated model | 85% |
| Reliability, metrics, drift, model card | Done | 100% |
| Portable deployment | Docker + compose done; local Docker unavailable here for manual run | 85% |
| Physical/edge realism | MQTT contract + optional broker publisher + TSDB export done | 78% |
| Demo packaging | Visual demo hub and reviewer walkthrough done; demo video still pending | 94% |

Overall: **about 96% complete as a reviewable demo**, and **about 90% complete as
a production-like robotics data platform**.

## What Is Already Demo-Ready

- `/demo`: visual demo hub that links the 3D twin, 2D control center, operations
  report, shift handover, model card, and visible artifacts in one entry point.
- `/twin`: 3D FAB/AGV digital twin with touch navigation and warning highlights.
- `/`: 2D control center for KPI, alert feed, and fleet status scanning.
- `/ws`: live fleet state stream.
- `/api/snapshot`: current AGV state, KPI, and per-asset PHM forecast contract.
- `/api/phm`: PHM forecast summary with stage, severity, risk score, RUL estimate,
  reasons, and recommended action.
- `/api/snapshot` inference block: live LightGBM Booster mode, feature count,
  latency, call count, and replay audit fields.
- `/api/edge-contract`, `/api/edge-events`: MQTT-compatible topic/payload
  schema and recent edge telemetry message buffer.
- `src/mqtt_bridge.py`: optional publisher that reads `/api/snapshot` and sends
  validated edge telemetry envelopes to a real MQTT broker, with dry-run support.
- `/api/history`, `/api/stats`, `/api/trend`: telemetry persistence and analysis.
- `/api/tsdb-contract`, `/api/tsdb-export?fmt=influx`,
  `/api/tsdb-export?fmt=timescale`: external time-series DB export path.
- `/api/reliability`: MTBF, MTTR, availability.
- `/api/work-orders`: predictive fault / low-health AGVs converted to P1-P3
  maintenance work orders with status tracking.
- `/api/data-quality`: robotics dataset QA/governance metrics.
- `/api/drift`: live data drift monitoring.
- `/api/model-card`: model artifact hash, feature contract, metrics, limitations.
- `/api/reviewer-brief`: 3-minute reviewer path, proof points, and role mapping.
- `/metrics`: Prometheus-compatible monitoring.
- `docker compose up --build`: one-command deployment entrypoint.

## Remaining Work

| Priority | Work | Why it matters |
|---|---|---|
| P1 | Demo video, 2-3 minutes | Makes the project instantly reviewable in portfolio/resume contexts. |
| P1 | README demo-video link and capture checklist | Turns the repository front page into a one-click visual review path. |
| P3 | Physical robot or MQTT-fed sensor windows | Replaces the deterministic simulator windows with a real edge source. |

## Recommended Next Three Daily Commits

1. `docs(demo): add video script and capture checklist`
2. `feat(edge): add mqtt-fed sensor replay input`
3. `docs(demo): add 2-minute capture script and shot list`
