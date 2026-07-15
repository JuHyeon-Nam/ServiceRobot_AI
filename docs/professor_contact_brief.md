# Professor Contact Brief

> 석사과정 컨택 메일에 첨부하거나 링크로 보낼 1페이지 연구 브리프입니다.  
> 전체 README보다 먼저 이 문서를 보여주면 연구 적합성을 빠르게 전달할 수 있습니다.

## Research Fit

저는 제조 현장 설비제어 경험과 소프트웨어/AI 구현 경험을 함께 가진 지원자로, 센서 시계열 기반 예지보전(PdM), 설비 건전성관리(PHM), 제조 공정 이상탐지, 디지털 트윈 기반 관제 연구에 관심이 있습니다.

이 프로젝트는 서비스 로봇 센서 데이터를 사용했지만, 핵심 문제 구조는 제조 설비와 같습니다.

- 다채널 센서 시계열에서 정상/고장 상태를 구분
- 희귀 고장과 정상 다수 클래스가 섞인 불균형 데이터 처리
- 처음 보는 장비에 대한 일반화 성능 검증
- 모델 판단 근거를 관제자가 이해할 수 있는 물리 신호로 설명
- 진단 결과를 실시간 관제, 이력 저장, 신뢰성 지표로 연결

## What I Built

| Area | Implementation |
|---|---|
| Data pipeline | AI-Hub 서비스 로봇 JSON 데이터에서 30시점 윈도우와 정적/맥락 피처 구성 |
| Feature engineering | Flatten, mean/std, drift, FFT 주파수 피처, 누적 주행거리/배터리/상태 피처 복원 |
| Model | LightGBM native model, 2.8MB, CPU 단독 1ms 추론 |
| Validation | 공식 validation split 93.4%, 랜덤분할 97.7%를 분리 기록 |
| Explainability | LightGBM contribution 기반 Top-3 물리 신호 설명 |
| System | FastAPI, WebSocket, Three.js 3D twin, SQLite event store, Prometheus metrics |
| Data governance | Robotics learning-data schema validation, annotation coverage, QA pass rate, rework rate |
| Operations | pytest contract tests, GitHub Actions CI, Docker smoke test |

## Research Questions I Want To Extend

1. **Domain generalization**  
   로봇/AGV/제조 설비처럼 장비가 달라져도 유지되는 고장 피처를 어떻게 학습할 수 있는가?

2. **Sensor-limited diagnosis**  
   특정 고장이 정상 노이즈와 중첩될 때, 추가 센서 또는 이종 데이터 융합이 어느 지점부터 성능을 개선하는가?

3. **Physics-informed feature design**  
   FFT, drift, cumulative usage 같은 물리적 의미가 있는 피처와 end-to-end 모델을 어떻게 결합할 것인가?

4. **Decision-oriented PHM**  
   단순 분류 결과를 Health Index, maintenance priority, MTBF/MTTR 같은 운영 의사결정 지표로 어떻게 변환할 것인가?

## Why This Matters For Manufacturing AI

현장 예지보전은 정확도 하나로 끝나지 않습니다. 실제 적용에는 센서 신뢰성, 희귀 고장, 오탐/미탐 비용, 설명가능성, 이력 저장, 정비 우선순위가 함께 필요합니다. 이 프로젝트는 그 전체 흐름을 작은 규모로 end-to-end 구현한 프로토타입입니다.

## Links

- GitHub: https://github.com/JuHyeon-Nam/ServiceRobot_AI
- Main demo route: `uvicorn realtime_server:app --reload` 후 `/twin`
- API route: `uvicorn app:app --reload` 후 `/docs`
