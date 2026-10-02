# ServiceRobot_AI

AI-Hub 서비스 로봇 센서 데이터를 이용해 **센서 구간 생성, 고장 진단, 추론 API,
실시간 상태 스트리밍, 2D·3D 관제**를 하나의 실행 흐름으로 연결한 PHM 프로젝트입니다.

[![CI](https://github.com/JuHyeon-Nam/ServiceRobot_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/JuHyeon-Nam/ServiceRobot_AI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-02569B?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?style=flat-square&logo=fastapi&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-3D%20monitoring-000000?style=flat-square&logo=threedotjs&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-telemetry-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white)

[GitHub 프로필](https://github.com/JuHyeon-Nam)

## Demo

`/twin`과 `/demo`는 같은 통합 관제 화면입니다. 자산 목록에서 정상 AGV도 선택할 수 있으며,
**센서·진단 확인 → PHM 위험 평가 → 정비 접수·점검·완료 → 이력·리포트**로 이어집니다.

![통합 관제 워크스페이스](assets/twin_workspace.png)

[통합 관제 시연 영상 (38초)](assets/twin_walkthrough.mp4) · [상세 시연 순서](docs/DEMO_CAPTURE_CHECKLIST.md)

기존 3D 구현 기록도 보존합니다.

![Replay-based 3D monitoring with live model inference](assets/twin_3d.gif)

| 2D control center | Feature importance |
|---|---|
| ![Control center](assets/control_center.png) | ![Feature importance](assets/feature_importance.png) |

| Per-class F1 | Confusion matrix |
|---|---|
| ![Per-class F1](assets/per_class_f1.png) | ![Confusion matrix](assets/confusion_matrix.png) |

## 관제 흐름

1. 자산 검색과 층·상태 필터로 AGV를 선택합니다. 선택한 층에 집중하고 정상·이상 자산 모두 확대합니다.
2. 정상은 초록, 주의는 노랑, 예측 이상은 주황, 현재 이상은 빨강으로 표시합니다.
3. 자산 상세의 시연 상태에서 `정상 → 주의 → 예측 이상 → 배터리 이상`을 재현합니다.
4. 정비 작업 탭에서 `접수 → 점검 시작 → 조치 완료 → 종결`을 처리합니다. 완료한 작업은 동일 신호로 중복 생성하지 않으며, 정상 복귀 후 재발하면 새 작업을 만듭니다.
5. 데이터·AI 탭에서 공식 검증 지표, 입력 출처, PHM의 검증 범위를 확인합니다.

기본 주행은 0.5×로 약 125초에 한 바퀴이며 0.25× / 0.5× / 1×와 주행 일시정지를 지원합니다.
이는 시각화 좌표의 시뮬레이션으로 실제 m/s 보정은 미적용입니다. 재생 설정과 시나리오는 같은 서버 세션에서 공유됩니다.

## Project Summary

| Item | Evidence |
|---|---|
| Data | AI-Hub 실내공간 유지관리 서비스 로봇 JSON, 100만 건 이상 |
| Task | 정상과 8개 고장을 구분하는 9-class 상태진단 |
| Input | 30 timesteps, 5개 동적 센서와 9개 context 변수 |
| Feature | flatten·평균·표준편차·drift·rFFT를 결합한 249개 특징 |
| Model | LightGBM native model, 4.30 MiB, best iteration 73 |
| Validation | official accuracy `0.9329`, macro-F1 `0.5838` |
| Serving | FastAPI REST API, WebSocket state stream |
| Interface | Canvas 2D 관제, Three.js 3D·자산 상세·정비·데이터/AI 통합 UI |
| Operations | SQLite history, data QA, drift, reliability, work-order APIs |

`0.9329`는 학습에 사용하지 않은 robot instance로 구성된 official validation split
결과입니다. 희귀 고장 클래스의 표본이 적어 macro-F1이 낮다는 점도 함께 공개합니다.

## Scope and Contribution

- 원본 JSON을 30시점 sensor window로 묶고 학습·검증용 array로 변환
- 통계·변화·주파수 특징을 구성하고 LightGBM 모델 학습·평가
- 학습과 서빙이 동일한 249개 feature contract를 사용하도록 모듈화
- `/predict` 추론 API와 `/ws` 상태 스트림 구성
- 2D 관제와 Three.js 3D twin에서 상태·경고·진단 근거 시각화
- telemetry 이력, drift·data quality·model card, 작업지시 흐름 구현
- 물리 로봇 연결 전환을 위한 MQTT-compatible payload와 edge ingest 경로 구성
- 자산 단위 holdout RUL 학습·독립 고장 사건 집계·시연 데이터 제외
- 반응형 UI, WebSocket 장애 시 HTTP 폴링, 실제 WebGL 픽셀 기반 브라우저 검증

## Architecture

```mermaid
flowchart LR
    raw["AI-Hub robot JSON"] --> build["30-step window<br/>feature builder"]
    build --> split["official train / validation"]
    split --> model["LightGBM<br/>9-class diagnosis"]
    model --> api["FastAPI /predict"]
    model --> runtime["live inference runtime"]
    runtime --> ws["WebSocket /ws"]
    ws --> ui["2D control center<br/>Three.js twin"]
    runtime --> store["SQLite telemetry"]
    store --> ops["history · trend · reliability<br/>drift · work orders"]
    edge["MQTT / physical adapter"] --> ingest["validated edge ingest"]
    ingest --> runtime
```

## Model Pipeline

### Input contract

- Raw dynamic sensors: `batteryLevel`, `speed`, `x`, `y`, `degree`, `collision`, `obstacle`
- Model dynamic sensors: `batteryLevel`, `speed`, `degree`, `collision`, `obstacle`
- Context: `isOffline`, `nowCharging`, `emergencyStop`, `batteryUse`,
  `batteryCycleCount`, `distance`, `crowd`, `deviceType`, `mainState`
- Output: normal state or one of eight fault codes with confidence

Absolute `x`, `y` coordinates are retained for visualization but excluded from model input.
This prevents the model from treating a site-specific coordinate as a shortcut for a fault.

### Diagnostic classes

| Code | Meaning |
|---|---|
| `normal` | 정상 |
| `E-ENV-C` | 충돌 위험 |
| `E-ENV-O` | 장애물·경로 방해 |
| `E-INF-A` | 자동문 인터페이스 이상 |
| `E-INF-E` | 엘리베이터 인터페이스 이상 |
| `E-RBT-B` | 배터리 이상 |
| `E-RBT-E` | 비상정지 |
| `E-RBT-N` | 네트워크 이상 |
| `E-RBT-S` | 센서 이상 |

## Troubleshooting

### 1. High accuracy hid rare-fault failures

정상 데이터가 약 83%인 환경에서는 accuracy만으로 모델을 평가하면 희귀 고장 미탐을
가릴 수 있었습니다. official split의 macro-F1과 클래스별 F1, confusion matrix를 함께
확인하고, validation support가 매우 작은 고장은 별도 한계로 표시했습니다.

### 2. Coordinates behaved like a site identifier

초기 분석에서 절대좌표가 높은 중요도를 보였습니다. 다른 사이트에서는 좌표계가 달라질
수 있으므로 `x`, `y`를 모델 입력에서 제거하고 이동 방향과 상태·누적 신호 중심으로
feature contract를 다시 구성했습니다.

### 3. Training and serving could diverge

학습 코드와 API가 각각 특징을 만들면 순서·차원이 어긋날 위험이 있었습니다.
`pdm_runtime.py`의 공통 feature builder와 model metadata 검증을 통해 249개 입력 순서를
고정하고, API contract test로 회귀를 확인했습니다.

### 4. A visual demo could be mistaken for a physical deployment

3D 이동과 sensor window는 재현 가능한 합성 입력이며, LightGBM은 미캐시 window에서 추론합니다.
수동 시나리오는 모델 출력을 강제한 시연 상태로, 화면에 신뢰도를 표시하지 않습니다. 이를 `/api/data-source`와 model card에 노출하고, 실제 센서는
`/api/edge-ingest` 앞단의 MQTT subscriber 또는 physical adapter로 교체하도록 경계를
분리했습니다.

## Real-time and Operations Layer

| Endpoint | Purpose |
|---|---|
| `POST /predict` | 30-step sensor window 상태진단 |
| `GET /api/snapshot` | AGV 상태·진단·경고 snapshot |
| `WS /ws` | 실시간 fleet state stream |
| `GET /api/history` | asset별 telemetry 이력 |
| `GET /api/trend` | 시간 구간별 상태 집계 |
| `GET /api/data-quality` | 입력 데이터 QA |
| `GET /api/drift` | 기준 운전분포 대비 drift 확인 |
| `GET /api/model-card` | artifact hash·feature contract·한계 |
| `GET /api/work-orders` | 이상 상태를 정비 작업 후보로 변환 |
| `POST /api/edge-ingest` | 검증된 외부 telemetry 입력 |
| `GET/POST /api/demo` | 자산별 시나리오와 재생 제어 |
| `POST /api/work-orders/{id}/status` | 정비 접수·점검·완료·종결 |
| `GET /api/rul-contract` | RUL 준비도·선택형 오프라인 학습 메타데이터 |
| `GET /api/shift-handover?fmt=md` | 인수인계 리포트 내보내기 |

PHM 위험도와 RUL은 현재 health·trend·sensor signal을 이용한 heuristic입니다.
실제 failure-time label이 확보될 때 supervised regression 또는 survival model로 교체할 수
있도록 dataset builder, model slot, 오프라인 baseline trainer를 구성했습니다. smoke fixture 결과를 현장 성능으로
주장하지 않습니다.

- `source`로 `replay_model`, `demo_scenario`, `edge_ingest`를 구분합니다. 외부 입력은 수동 시나리오보다 우선합니다.
- 이벤트 이력·CSV·TSDB export에 출처를 남기고, MQTT 왕복에서도 수동 시연 출처를 유지합니다.
- RUL dataset builder는 수동 시연 기록을 제외합니다. 준비도는 센서 행 수가 아니라 독립 자산·고장 시각 쌍으로 계산합니다.
- RUL trainer는 자산별 holdout으로 분리하며, 단일 자산 데이터의 학습/평가 재사용을 거부합니다.
- 선택형 `RUL_BASELINE_META` 메타데이터가 있어도 운영 PHM 모델로 자동 전환하지 않습니다.
- 3D 점검 근거는 코드별 점검 항목과 PHM 규칙 신호입니다. 모델 기여도 분석은 별도 `/predict` 경로에서 제공합니다.
- 이벤트 기반 MTBF/MTTR·가용도·영향도는 시뮬레이션 관측 지표이며 검증된 생산 KPI가 아닙니다.

## Run

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-server.txt
cd src
uvicorn realtime_server:app --reload
```

- 통합 관제: `http://127.0.0.1:8000/demo`
- 2D control center: `http://127.0.0.1:8000/`
- 3D twin: `http://127.0.0.1:8000/twin`
- API docs: `http://127.0.0.1:8000/docs`

센서 window를 직접 전송하는 `/predict` 및 기여도 기반 설명은 별도 추론 서버에서 제공합니다.
새 터미널에서 `cd src` 후 `uvicorn app:app --port 8001`을 실행하면
`http://127.0.0.1:8001/docs`에서 입력·응답을 확인할 수 있습니다.

### Docker

```bash
docker compose up --build
```

Optional MQTT smoke profile:

```bash
docker compose --profile mqtt-smoke up --build
```

## Test

백엔드 검증:

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m compileall src tests
```

별도 로컬 시연 서버를 실행한 뒤 브라우저 검증:

```bash
npm ci
npx playwright install chromium
CHROME_CHANNEL=chromium TWIN_URL=http://127.0.0.1:8000 npm run test:ui
```

브라우저 검증은 6개 화면 크기, 정상·주의·예측·고장 시나리오, 정비 처리,
WebGL 렌더링·클릭·움직임, WebSocket 장애 시 폴링 복구를 확인합니다.
설정은 복원하지만 테스트 작업 이력은 남으므로 별도 시연 서버에서 실행합니다.
기존 GitHub Actions는 Python·Docker 검증을 수행합니다. 브라우저 검증을 추가하는
[워크플로 예제](docs/ci-workflow-with-browser.example.yml)도 제공합니다.
현재 푸시 토큰에 `workflow` 권한이 없어 자동 등록은 보류되어 있으며, 브라우저 검증은 로컬에서 실행했습니다.

Tests cover feature contracts, prediction, twin payloads, telemetry storage, edge ingest,
data QA, drift, reliability, work orders and RUL dataset preparation.

## Repository Guide

| Path | Description |
|---|---|
| `src/build_enhanced_dataset.py` | JSON parsing and window generation |
| `src/train_enhanced.py` | LightGBM training |
| `src/evaluate_enhanced.py` | official validation evaluation |
| `src/pdm_runtime.py` | shared feature and inference contract |
| `src/app.py` | inference API |
| `src/realtime_server.py` | WebSocket, twin and operations API |
| `src/telemetry_store.py` | SQLite telemetry layer |
| `src/static/` | 2D 관제, 통합 3D UI, 로컬 Three.js·Lucide |
| `src/demo_runtime.py` | 명시적 시연 상태 및 제어 계약 |
| `scripts/verify_twin.mjs` | 반응형 UI·WebGL·정비 흐름 검증 |
| `scripts/capture_demo.mjs` | 실제 브라우저 시연 프레임 캡처 |
| `docs/MODEL_CARD.md` | metrics, artifact and limitations |
| `docs/PROJECT_STATUS.md` | implemented scope and remaining work |

## Limitations

- Training data is a public service-robot dataset, not a production FAB dataset.
- The 3D twin uses replay trajectories and generated runtime windows.
- Rare classes have low validation support and require additional data collection.
- Physical robot, real broker and external time-series database integration require field validation.
- RUL is not a field-calibrated remaining-life model.

The repository is designed to keep those boundaries visible while showing an end-to-end path from
sensor data to diagnosis, monitoring and maintenance action.
