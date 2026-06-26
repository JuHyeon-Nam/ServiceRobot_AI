# 🧭 혼자 이어서 개발하기 — 업그레이드 가이드

> 나중에 이 프로젝트를 다시 열었을 때 "뭐부터 어떻게"를 잃지 않도록 정리한 문서.
> 위에서 아래로 우선순위 순. 각 단계는 **독립적**이라 원하는 것부터 골라 해도 됨.

---

## 0. 다시 시작하기 (어느 컴퓨터든)

```bash
git clone https://github.com/JuHyeon-Nam/ServiceRobot_AI.git
cd ServiceRobot_AI
python -m venv venv && venv\Scripts\activate      # mac/linux: source venv/bin/activate
pip install -r requirements.txt

cd src
python evaluate_enhanced.py      # 모델 동작 확인 (정확도 93.3% 나오면 정상)
streamlit run dashboard.py       # 관제 대시보드 실행
```
- **모델·재생데이터는 repo에 포함** → 위만 하면 바로 돌아감(데이터셋 불필요).
- **재학습/재추출**이 필요할 때만 원본 4.2GB가 필요. `build_enhanced_dataset.py` 맨 위 `BASE` 경로를 새 위치로 바꾸고 실행 → `train_enhanced.py`.

---

## ✅ 지금까지 완성된 것
데이터 파이프라인 · 경량 모델(2.8MB) · 정직한 평가(공식 93.3%) · FastAPI 서빙 · 디지털 트윈 관제 대시보드 · 시각화 4종.
→ **핵심 기능은 끝.** 아래는 "더 고퀄·더 진짜 시스템"으로 키우는 단계.

---

## STEP 1 — 진짜 실시간 스트리밍 ⭐ (가장 임팩트 큼)
**목표:** 지금은 녹화 재생. 이걸 "센서가 실시간으로 흘러들어오는 진짜 관제"로 격상.

**어떻게:**
1. `simulator.py` 작성 — `replay.parquet`을 0.5초 간격으로 한 프레임씩 흘려보냄.
2. `app.py`에 WebSocket 엔드포인트 추가(`/ws`) — 시뮬레이터가 보낸 센서를 모델로 진단해 브로드캐스트.
3. `dashboard.py`가 WebSocket을 구독해 실시간 렌더(현재의 사전계산 대신).
   - 라이브러리: `websockets` 또는 `paho-mqtt`(+ mosquitto 브로커).
4. (선택) **MQTT**로 바꾸면 프로필의 *Edge-Cloud/MQTT* 주장과 정확히 연결됨.

**왜:** "실시간 관제"가 말이 아니라 실제가 됨. 면접에서 가장 강한 데모.
**난이도:** 중 · **예상:** 3~5일

---

## STEP 2 — Docker 패키징
**목표:** `docker compose up` 한 줄로 API+대시보드 실행.

**어떻게:**
1. `Dockerfile` (python:3.11-slim 베이스, requirements 설치).
2. `docker-compose.yml` — api(FastAPI) + dashboard(Streamlit) 두 서비스.
3. README에 `docker compose up` 실행법 추가.

**왜:** "배포까지 안다"는 신호 + 어느 환경에서도 동일 실행(재현성).
**난이도:** 하 · **예상:** 1~2일

---

## STEP 3 — 모델 설명가능성 (SHAP)
**목표:** "왜 배터리 저하라고 판단했나"의 근거를 보여줌.

**어떻게:**
1. `pip install shap`, `shap.TreeExplainer(booster)`.
2. 예측 1건의 기여 피처 Top3를 추출(`distance ↑`, `batteryUse ↑` 등).
3. `app.py` 응답과 대시보드 경고에 "근거: distance↑, batteryUse↑" 표시.

**왜:** PdM에서 "왜"는 핵심. AI를 블랙박스로 안 쓴다는 깊이.
**난이도:** 중 · **예상:** 2~3일

---

## STEP 4 — DB 연동 (이력·감사)
**목표:** 센서 이력·예측·알림을 저장하고 조회.

**어떻게:**
1. SQLite(간단) 또는 PostgreSQL/TimescaleDB(시계열 특화).
2. `SQLAlchemy` 모델: `readings`, `predictions`, `alerts` 테이블.
3. `app.py`가 매 예측을 INSERT, 대시보드에 "최근 알림 이력" 조회 패널 추가.

**왜:** 데이터 엔지니어 차원 추가(수집→저장→조회 파이프라인).
**난이도:** 중 · **예상:** 2~4일

---

## STEP 5 — 견고함 (테스트 · CI)
**목표:** 코드 신뢰성 + 깃허브 자동 검증 배지.

**어떻게:**
1. `tests/` — `pytest`로 `create_features` 차원·`/predict` 응답 형태 검증.
2. `.github/workflows/ci.yml` — push 시 테스트 자동 실행.
3. README에 CI 배지 추가.

**왜:** 협업·실무 역량 신호. 적은 노력 대비 신뢰도↑.
**난이도:** 하 · **예상:** 1~2일

---

## STEP 6 — 욕심내면 (스트레치)
- **ONNX 변환** — `robot_pdm_enhanced.txt`→ONNX, 엣지/모바일/타 언어 추론 + 지연 벤치마크.
- **데이터 드리프트 감지** — 입력 분포가 학습과 벌어지면 경고(실서비스 PdM 핵심).
- **MLflow** — 실험·모델 버전 추적 + 모델 카드.
- **데모 영상(2~3분)** — 대시보드 시연 녹화, README/이력서에 링크.

---

## 작업 루틴 (막히지 않게)
1. 한 번에 **STEP 하나만**. 끝나면 커밋·푸시.
2. 커밋 author는 항상 본인(`남주현 / nam3847@inha.edu`). co-author 트레일러 금지.
3. 각 STEP 끝나면 README 로드맵 체크박스 갱신 + 필요한 시각화/스크린샷 추가.
4. 막히면: 해당 STEP만 떼어내 작은 예제로 먼저 동작 확인 → 본 프로젝트에 통합.

**추천 순서:** STEP 1(실시간) → STEP 2(Docker) → STEP 3(SHAP). 이 셋이면 "완성도 높은 실시간 AI 시스템"으로 충분.
