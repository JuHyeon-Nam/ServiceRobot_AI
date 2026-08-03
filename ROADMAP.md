# 🗺️ ServiceRobot_AI — 마스터 포트폴리오 로드맵

> 목표: 이 프로젝트를 **"태블릿으로 조작하는 실시간 3D 디지털 트윈 + 예지보전(PdM) 관제 시스템"**으로 완성한다.
> 두 마리 토끼: ① **반도체 설비기술(무인화·관제 SW)·데이터 직무 마스터 포폴**, ② **2026년 11월 졸업작품 전시 데모**.

---

## 🎯 비전 (한 문장)

> 현장 3D 도면 위에서 설비·AGV가 실시간으로 움직이고, AI가 이상을 즉시 감지·하이라이트하며,
> **관람자가 태블릿으로 도면을 직접 조작(회전·확대·설비 선택)**하면 해당 설비의 실시간 센서·AI 진단·근거가 뜨는 **디지털 트윈 관제**.

타겟 직무 직결: 삼성 DS 설비기술 직무의 *"생산 무인화 시스템·시스템 관제·물류 반송(AGV/AMHS)"* + 데이터·AI(PdM).

---

## ✅ 이미 완성 (Baseline)
- 경량 PdM 모델(LightGBM 2.8MB, 공식 Val 93.3%) + 추론 근거(SHAP) `/predict`
- FastAPI 추론 API + pytest
- **2D** 실시간 관제: `realtime_server.py`(FastAPI+WebSocket `/ws`) + `static/index.html`(Canvas) + `fab_layout.py`(층/장비/트랙/AGV 단일 소스)

---

## 🚀 마스터 빌드 백로그 (매일 최대한, 일정 안 쪼갬 — 하나씩 완성→push)

### A. 3D 디지털 트윈 (핵심)
- [x] **A1 3D 트윈 기반** — Three.js 씬, 3개 층 슬래브, 장비 3D 박스, AGV 3D 이동, 태블릿 OrbitControls, 탭→설비 AI 진단 패널 ✅ (`/twin`)
- [x] A2 OHT 오버헤드 레일 + 스토커 타워 + 층간 리프트(캐빈) 3D 구조물 ✅
- [x] A3 이상 설비 3D 하이라이트(발광·경고 링·층관통 경고빔·맥동) + 최우선 경고 카메라 자동 포커스 ✅
- [x] A4 설비 탭 시 **실시간 센서 그래프**(진동·배터리·온도 스파크라인) + AI 판단 근거(주요 이상 신호) 표시 ✅
- [x] A5 태블릿 UX 마감 — 더블탭 시점 리셋 제스처, 풀스크린 토글, 세로/좁은 화면 반응형(패널 축소·큰 터치 타깃) ✅
- [x] A6 **오프라인 동작**(Three.js 로컬 벤더링) — 전시장 인터넷 없어도 구동 ✅

### B. 실시간 감지 엔진 고도화
- [x] B1 서버가 매 tick 실제 `booster.predict` 호출 — `/api/snapshot` live Booster inference + latency/metrics/audit 필드 ✅
- [x] B2 드리프트/이상 누적 감지 + 경고 등급(주의/경고/위험) — 경고 등급(신뢰도 기반 3단계 트리아지·KPI 집계·UI 배지) ✅ + **추세 방향(악화/개선/안정) 드리프트 조기 감지**(정상이어도 악화 중이면 경고, 플릿 악화 대수 집계) ✅
- [x] B3 설비별 이력 타임라인 — 설비 탭 시 최근 12틱 진단 추세를 색상 셀 타임라인으로(정상/주의/경고/위험) ✅

### C. 시스템 완성도
- [x] C1 테스트 보강(서버 스냅샷·레이아웃 계약) + **GitHub Actions CI**(push마다 pytest 자동 실행 + README 배지) ✅
- [x] C2 Docker `compose up` 한 줄 실행 — durable telemetry volume + healthcheck ✅
- [ ] C3 데모 영상 2~3분
- [x] C4 데이터 드리프트 감지 — 실시간 입력 분포를 기준 운전 프로파일과 비교(`/api/drift`, Prometheus 게이지) ✅
- [x] C5 모델 카드/모델 거버넌스 — artifact SHA256·피처 계약·성능·한계·재학습 트리거 문서/API화 ✅
- [x] C6 포트폴리오 리뷰어 가이드 — 3분 시연 순서·직무 키워드 매핑·면접 포인트 문서/API화 ✅
- [x] C7 예측정비 작업지시 — AI 경고·저건전도 설비를 P1/P2/P3 작업지시로 자동 생성하고 상태 추적(`/api/work-orders`) ✅
- [x] C8 MQTT-style 엣지 수집 계약 — AGV별 topic/payload schema와 최근 edge message buffer(`/api/edge-contract`, `/api/edge-events`) ✅
- [ ] C9 (스트레치) 실제 MQTT 브로커 수집 / 외부 시계열DB 이력

---

## 🔁 작업 원칙
1. 매일 **할 수 있는 만큼 최대한**, 끝낸 조각은 바로 커밋·push(잔디는 진짜 작업).
2. 커밋 author = `남주현 <nam3847@inha.edu>`, **Claude/AI 트레일러 금지.**
3. 깨진 코드 push 금지 — 테스트 통과 후 커밋. `DEVLOG.md`·체크박스 갱신.
4. 우선순위: A(3D 트윈) → B(감지) → C(완성도). 전시(11월) 역산해 A를 먼저 굳힌다.

> 📌 취업 일정·진로: `../JuHyeon-Nam/진로컨설팅/` · 단계 가이드: [NEXT_STEPS.md](NEXT_STEPS.md)
