# Project Status — ServiceRobot_AI

## Ultimate Target

The final portfolio artifact is a **tablet-operable real-time 3D digital twin
and predictive-maintenance control center** for service robots / FAB AGV fleets.

It should show five capabilities in one coherent demo:

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
| 3D digital twin demo | Done; slower replay, light high-legibility UI, normal/warning AGV click focus | 100% |
| Live inference in twin stream | Done | 100% |
| MQTT-style edge telemetry contract | Done | 100% |
| Optional MQTT broker bridge | Done; publisher/subscriber runners plus Mosquitto compose smoke profile | 100% |
| MQTT-fed sensor replay input | Done; inbound broker/API/physical-adapter payload overrides twin snapshot for a short TTL | 100% |
| Telemetry storage / history / rollup | Done; stores PHM risk score and trend slope for RUL training | 100% |
| External TSDB export contract | Done; Influx line protocol and Timescale SQL export include PHM/RUL features | 90% |
| Predictive maintenance work orders | Done | 100% |
| Operations dispatch plan | Done; per-AGV impact, SLA, route-block risk, and work-order candidate | 100% |
| PHM forecast contract | Done; heuristic RUL includes supervised model slot, calibration contract, dataset builder, and offline baseline trainer | 97% |
| Reliability, metrics, drift, model card | Done | 100% |
| Portable deployment | Docker + compose + MQTT profile done; local Docker CLI unavailable here for manual run | 90% |
| Physical/edge realism | MQTT contract + publisher + subscriber + Mosquitto profile + physical sensor adapter + TSDB export done | 96% |
| Demo packaging | Visual demo hub, reviewer walkthrough, and capture checklist done; demo video still pending | 96% |

Overall: **about 99% complete as a reviewable demo**, and **about 97% complete as
a production-like robotics data platform**.

## What Is Already Demo-Ready

- `/demo`: visual demo hub that links the 3D twin, 2D control center, operations
  report, shift handover, model card, and visible artifacts in one entry point.
- `/twin`: 3D FAB/AGV digital twin with touch navigation and warning highlights.
- `/`: 2D control center for KPI, alert feed, and fleet status scanning.
- `/ws`: live fleet state stream.
- `/api/snapshot`: current AGV state, KPI, and per-asset PHM forecast contract.
- `/api/data-source`: explicit disclosure of AI-Hub replay, live LightGBM inference,
  rule-based PHM/RUL, Edge TTL input, and physical-robot connection status.
- `/api/phm`: PHM forecast summary with stage, severity, risk score, RUL estimate,
  reasons, recommended action, and a transparent RUL model slot.
- `/api/rul-contract`: feature fields, failure-time label requirements, readiness
  checks, and sample AGV feature vectors for replacing heuristic RUL with a
  calibrated regression or survival model.
- `/api/dispatch-plan`: converts model/PHM output into operations impact,
  affected zone, route-block risk, priority/SLA, and work-order candidate.
- `/api/snapshot` inference block: live LightGBM Booster mode, feature count,
  latency, call count, and replay audit fields.
- `/api/edge-contract`, `/api/edge-events`: MQTT-compatible topic/payload
  schema and recent edge telemetry message buffer.
- `src/mqtt_bridge.py`: optional publisher that reads `/api/snapshot` and sends
  validated edge telemetry envelopes to a real MQTT broker, with dry-run support.
- `src/mqtt_subscriber.py`: optional subscriber that receives broker telemetry,
  validates the payload contract, and forwards it into `/api/edge-ingest`.
- `src/physical_sensor_adapter.py`: JSON/CSV sensor-line adapter that normalizes
  physical edge readings into the same `/api/edge-ingest` contract.
- `docker compose --profile mqtt up --build`: starts the API, Mosquitto broker,
  and subscriber; `--profile mqtt-smoke` runs a one-shot publisher smoke.
- `/api/edge-ingest`: validates inbound edge/MQTT payloads and temporarily
  applies them to `/api/snapshot`, `/twin`, and PHM forecast output.
- `/api/history`, `/api/stats`, `/api/trend`: telemetry persistence and analysis,
  including risk score and trend slope needed by RUL training.
- `/api/tsdb-contract`, `/api/tsdb-export?fmt=influx`,
  `/api/tsdb-export?fmt=timescale`: external time-series DB export path with
  PHM/RUL feature fields.
- `src/build_rul_dataset.py`: joins stored telemetry with real failure labels to
  produce a supervised-ready RUL training table.
- `src/train_rul_baseline.py`: trains a Gradient Boosting RUL regression baseline
  from observed failure rows and reports median-baseline comparison metrics.
- `/api/reliability`: MTBF, MTTR, availability.
- `/api/work-orders`: predictive fault / low-health AGVs converted to P1-P3
  maintenance work orders with status tracking.
- `/api/data-quality`: robotics dataset QA/governance metrics.
- `/api/drift`: live data drift monitoring.
- `/api/model-card`: model artifact hash, feature contract, metrics, limitations.
- `/api/reviewer-brief`: 3-minute reviewer path, proof points, and role mapping.
- `docs/DEMO_CAPTURE_CHECKLIST.md`: 2-3 minute screen-recording shot list.
- `/metrics`: Prometheus-compatible monitoring.
- `docker compose up --build`: one-command deployment entrypoint.

## Remaining Work

| Priority | Work | Why it matters |
|---|---|---|
| P1 | Demo video, 2-3 minutes | Makes the project instantly reviewable in portfolio/resume contexts. |
| P1 | README demo-video link | Turns the repository front page into a one-click visual review path. |
| P3 | Calibrated RUL model | Replace the smoke-trained baseline with enough real failure-time labels, then validate regression or survival modeling. |

## Recommended Next Three Daily Commits

1. `docs(demo): add final video link placeholder`
2. `feat(phm): expose optional trained RUL artifact metadata in /api/rul-contract`
3. `feat(phm): add survival-model path for right-censored rows`
