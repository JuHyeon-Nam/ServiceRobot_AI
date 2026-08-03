# Daily Sprint Plan

> 매일 잔디를 채우기 위한 문서가 아니라, 매일 프로젝트가 실제로 좋아지도록 만드는 작은 스프린트 목록입니다.  
> 원칙: 하루에 하나만 끝내고, 테스트 또는 문서 증거를 남긴 뒤 커밋합니다.

## Current Position

이미 완성된 핵심:

- LightGBM 기반 PdM 모델과 공식 validation 평가
- FastAPI 추론 API
- WebSocket 기반 실시간 관제 서버
- Three.js 3D 디지털 트윈
- MQTT-style edge telemetry topic/payload contract
- 시계열 이벤트 저장/집계/CSV 반출
- 예측정비 작업지시 API
- MTBF/MTTR/Availability, Prometheus metrics
- pytest contract tests, GitHub Actions CI, Docker smoke test

앞으로의 매일 작업은 "새 기능을 많이 붙이기"보다 **연구실 컨택과 면접에서 설명 가능한 깊이**를 쌓는 쪽으로 진행합니다.

## 14-Day Upgrade Queue

| Day | Task | Output |
|---:|---|---|
| 1 | README 숫자/표현 정합성 점검 | 성능 수치와 검증 기준 불일치 제거 |
| 2 | 교수 컨택용 1페이지 브리프 작성 | `docs/professor_contact_brief.md` |
| 3 | 모델 카드 작성 | 데이터, 모델, 평가, 한계, 윤리/적용 범위 정리 |
| 4 | 실험 로그 템플릿 추가 | feature/model/metric/lesson 기록 양식 |
| 5 | `realtime_server.py` endpoint 표 문서화 | API contract table |
| 6 | Health Index 산식 설명 문서화 | PHM 관점 설명 보강 |
| 7 | 센서 한계 분석 문서화 | 정상 노이즈와 희귀 고장 중첩 사례 정리 |
| 8 | MQTT-style edge telemetry 계약 구현 | `/api/edge-contract`, `/api/edge-events` |
| 9 | 외부 TSDB 확장 설계 | SQLite -> InfluxDB/TimescaleDB migration note |
| 10 | 실제 추론 tick 전환 계획 | 사전계산 replay -> live booster call risk list |
| 11 | Docker compose 추가 | `docker compose up` 한 줄 실행 |
| 12 | API 예시 curl 스크립트 추가 | 교수/면접 시연 재현성 향상 |
| 13 | 2분 데모 스크립트 작성 | 발표/면접용 demo talk track |
| 14 | 전체 문서 링크 정리 | README에서 컨택/시연/연구문서로 바로 이동 |

## Commit Rule

좋은 커밋 단위:

- `docs(contact): add professor research brief`
- `docs(phm): explain health index and maintenance priority`
- `test(api): lock reliability metrics contract`
- `feat(demo): add docker compose entrypoint`

피해야 할 커밋:

- 의미 없는 공백/날짜만 바꾸기
- 테스트가 깨진 상태의 기능 추가
- 과장된 성능 문구 추가

## Weekly Review Checklist

- README의 성능 숫자가 코드/메타파일과 일치하는가?
- 새로 추가한 기능이 연구 질문과 연결되는가?
- 컨택 메일에 보낼 링크가 너무 많지 않은가?
- 교수님이 2분 안에 "이 학생이 뭘 연구하고 싶은지" 이해할 수 있는가?
- 대기업 면접관이 "현장 적용 가능성"을 볼 수 있는가?
