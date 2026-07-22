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

## STEP 1 — 진짜 실시간 스트리밍 ⭐  ✅ 기본 구현 완료
**구현됨:** `realtime_server.py`(FastAPI + WebSocket `/ws`)가 모델 진단을 실시간 push,
`static/index.html`(Canvas)가 구독해 AGV를 라이브 렌더. `fab_layout.py`가 도면/경로 단일 소스.
```bash
cd src && uvicorn realtime_server:app --reload   # http://127.0.0.1:8000
```
**다음 고도화(선택):**
1. `simulator.py` 분리 — 별도 프로세스가 센서를 흘려보내고 서버는 받기만(진짜 수집-추론 분리).
2. 서버에서 매 tick `booster.predict`를 직접 호출(현재는 사전계산 진단 스트리밍) → 완전 라이브 추론.
3. **MQTT**(paho-mqtt + mosquitto)로 전송 계층 교체 → 프로필의 *Edge-Cloud/MQTT*와 직결.
4. 다중 클라이언트 브로드캐스트 + 재접속/백프레셔 처리.

**왜:** "실시간 관제"가 말이 아니라 실제가 됨. 면접에서 가장 강한 데모.

---

## STEP 2 — Docker 패키징 ✅ 기본 구현 완료
**구현됨:** `Dockerfile` + 서버 전용 `requirements-server.txt`로 관제 서버 이미지를 만들고, CI가 매 push마다 빌드·컨테이너 기동·엔드포인트 스모크 테스트를 수행.

```bash
docker build -t servicerobot-ai . && docker run -p 8000:8000 servicerobot-ai
```

**다음 고도화:** `docker-compose.yml`을 추가해 API/관제/모니터링을 더 명확히 분리.

**어떻게:**
1. `docker-compose.yml` — api(FastAPI) + optional monitoring service.
2. `TELEMETRY_DB` volume mount로 이벤트 이력 durable 처리.
3. README에 `docker compose up` 실행법 추가.

**왜:** "배포까지 안다"는 신호 + 어느 환경에서도 동일 실행(재현성).
**난이도:** 하 · **예상:** 1~2일

---

## STEP 3 — 모델 설명가능성 ✅ 기본 구현 완료
**구현됨:** `/predict`가 LightGBM contribution 기반 진단 근거 Top3를 반환하고, 관제 패널은 물리 신호 기반 원인 문구를 함께 표시.

**다음 고도화:**
1. 설명 결과를 모델 카드에 정리.
2. 정상/고장 샘플별 Top feature 비교표 추가.
3. 센서 한계 구간의 설명가능성 실패 사례를 별도 문서화.

**왜:** PdM에서 "왜"는 핵심. AI를 블랙박스로 안 쓴다는 깊이.
**난이도:** 중 · **예상:** 2~3일

---

## STEP 4 — DB 연동 (이력·감사) ✅ 기본 구현 완료
**구현됨:** `telemetry_store.py`가 SQLite 기반으로 진단 이벤트를 선별 적재하고, `/api/stats`, `/api/history`, `/api/trend`, CSV export를 제공.

**다음 고도화:**
1. InfluxDB 또는 TimescaleDB 확장 설계 문서화.
2. MQTT 수집 이벤트와 동일 인터페이스로 연결.
3. 장기 보존/다운샘플링 정책을 모델 카드에 명시.

**왜:** 데이터 엔지니어 차원 추가(수집→저장→조회 파이프라인).
**난이도:** 중 · **예상:** 2~4일

---

## STEP 5 — 견고함 (테스트 · CI) ✅ 기본 구현 완료
**구현됨:** pytest contract tests와 GitHub Actions CI가 push/PR마다 실행. Docker 이미지 빌드와 컨테이너 smoke test도 CI에 포함.

**다음 고도화:**
1. 실제 모델 메타파일과 README 수치 정합성 테스트.
2. API schema snapshot test.
3. Docker compose smoke test.

**왜:** 협업·실무 역량 신호. 적은 노력 대비 신뢰도↑.
**난이도:** 하 · **예상:** 1~2일

---

## STEP 6 — 욕심내면 (스트레치)
- **ONNX 변환** — `robot_pdm_enhanced.txt`→ONNX, 엣지/모바일/타 언어 추론 + 지연 벤치마크.
- ✅ **데이터 드리프트 감지** — `/api/drift`가 실시간 입력 분포와 기준 운전 프로파일을 비교해 watch/drift 등급과 재보정 권고를 반환.
- **MLflow** — 실험·모델 버전 추적 + 모델 카드.
- **데모 영상(2~3분)** — 대시보드 시연 녹화, README/이력서에 링크.

---

## 작업 루틴 (막히지 않게)
1. 한 번에 **STEP 하나만**. 끝나면 커밋·푸시.
2. 커밋 author는 항상 본인(`남주현 / nam3847@inha.edu`). co-author 트레일러 금지.
3. 각 STEP 끝나면 README 로드맵 체크박스 갱신 + 필요한 시각화/스크린샷 추가.
4. 막히면: 해당 STEP만 떼어내 작은 예제로 먼저 동작 확인 → 본 프로젝트에 통합.

**추천 순서:** 이제는 기능 추가보다 [교수 컨택용 연구 브리프](docs/professor_contact_brief.md), 모델 카드, 센서 한계 분석, MQTT/시계열DB 확장 설계처럼 "연구 적합성을 설명하는 문서"를 우선한다.
