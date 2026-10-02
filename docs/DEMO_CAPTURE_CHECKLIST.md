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
| 0:00-0:15 | `/demo` or `/twin` | Same integrated workspace, fleet summary, input-source labels |
| 0:15-0:55 | `/twin` | 3-floor FAB, slow motion, click a normal AGV, selected-floor focus |
| 0:55-1:25 | Selected AGV | Apply normal → watch → degrading → battery; identify scripted states, PHM heuristic, sensor traces |
| 1:25-1:45 | terminal + `/twin` | Physical sensor adapter line changes one AGV through `/api/edge-ingest` |
| 1:45-2:05 | Maintenance tab | Acknowledge → start inspection → resolve; return asset to normal |
| 2:05-2:25 | Data/AI tab | Official validation, macro-F1, input boundaries and uncalibrated PHM |
| 2:25-2:45 | `/api/rul-contract`, `/api/model-card` | Asset holdout, independent failures, optional offline metadata, artifact hash |
| 2:45-3:00 | README or `/api/data-source` | Data source boundary and production scale-up path |

## Reviewer Points

- Model replay runs a LightGBM Booster. Scripted scenario states and external reported diagnoses are explicitly labeled separately.
- PHM/RUL is currently a transparent heuristic, and `/api/rul-contract` exposes the feature/label/readiness contract for a calibrated RUL or survival model.
- `src/build_rul_dataset.py` can join telemetry events with future failure labels into a supervised RUL training table.
- `src/train_rul_baseline.py` turns that table into an offline RUL regression baseline and reports median-baseline comparison metrics.
- External input is already represented through `/api/edge-ingest`, MQTT publisher/subscriber, and `physical_sensor_adapter.py`.
- The 3D twin is connected to operations: dispatch priority, SLA, work orders, reliability, telemetry history, and reports.
- Data governance is visible through model card, drift, data-quality, and Prometheus metrics.

## Final README Placement

Keep the existing `assets/twin_3d.gif` as a historical implementation record.
Capture the upgraded workspace with `scripts/verify_twin.mjs`; it also exercises
the UI workflow and checks real WebGL pixels on six viewport sizes.

## 자동 캡처

Node.js 20 이상, `npm ci`, Chrome 또는 `npx playwright install chromium`, FFmpeg가 필요합니다.
별도 시연 서버를 켠 상태에서 실행합니다. 시나리오는 복원되지만 정비 이력은 남습니다.

```bash
CHROME_CHANNEL=chromium TWIN_URL=http://127.0.0.1:8000 npm run capture:demo
ffmpeg -framerate 5 -i /tmp/servicerobot-capture/frame-%04d.png \
  -vf fps=15 -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart \
  assets/twin_walkthrough.mp4
```

기본 캡처는 38초, 실제 브라우저 화면 190프레임입니다. 내레이션이 포함된 최종 2~3분 발표 영상은 별도로 제작합니다.
