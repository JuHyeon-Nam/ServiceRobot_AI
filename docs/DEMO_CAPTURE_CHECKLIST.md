# Demo Capture Checklist

## Goal

Record a 2-3 minute review video that shows ServiceRobot_AI as a working
real-time PdM/digital-twin system, not a static model notebook.

## Setup

```bash
cd src
uvicorn realtime_server:app --reload
```

Open:

- `http://127.0.0.1:8000/demo`
- `http://127.0.0.1:8000/twin`
- `http://127.0.0.1:8000/api/ops-report?fmt=md`
- `http://127.0.0.1:8000/api/model-card`

Optional edge smoke:

```bash
printf '{"asset_id":"AGV-01","floor":0,"vib":6.2,"batt":42,"temp":61}\n' \
  | python src/physical_sensor_adapter.py --ingest-url http://127.0.0.1:8000/api/edge-ingest
```

## Shot List

| Time | Screen | Show |
|---|---|---|
| 0:00-0:15 | `/demo` | System scope, demo entry points, artifacts |
| 0:15-0:55 | `/twin` | 3-floor FAB, slow AGV motion, status colors, warning focus |
| 0:55-1:25 | `/twin` selected AGV | Sensors, diagnosis, PHM, dispatch, work-order candidate |
| 1:25-1:45 | terminal + `/twin` | Physical sensor adapter line changes one AGV through `/api/edge-ingest` |
| 1:45-2:05 | `/api/rul-contract`, `/api/work-orders`, `/api/dispatch-plan` | RUL model-readiness boundary, maintenance priority, SLA |
| 2:05-2:25 | `/api/drift`, `/api/data-quality` | AI/data operations monitoring |
| 2:25-2:45 | `/api/model-card` | Artifact hash, feature contract, limitation disclosure |
| 2:45-3:00 | README or `/api/data-source` | Data source boundary and production scale-up path |

## Reviewer Points

- The live fault diagnosis path runs a LightGBM Booster, not a UI-only rule.
- PHM/RUL is currently a transparent heuristic, and `/api/rul-contract` exposes the feature/label/readiness contract for a calibrated RUL or survival model.
- `src/build_rul_dataset.py` can join telemetry events with future failure labels into a supervised RUL training table.
- External input is already represented through `/api/edge-ingest`, MQTT publisher/subscriber, and `physical_sensor_adapter.py`.
- The 3D twin is connected to operations: dispatch priority, SLA, work orders, reliability, telemetry history, and reports.
- Data governance is visible through model card, drift, data-quality, and Prometheus metrics.

## Final README Placement

After recording, add the video link near the top of `README.md` under `## 구현 화면`.
