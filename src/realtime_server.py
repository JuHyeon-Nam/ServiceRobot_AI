"""
realtime_server.py — 실시간 AGV 관제 서버 (FastAPI + WebSocket)
- 학습된 PdM 모델의 진단을 AGV 플릿 상태로 실시간 스트리밍.
- AGV 위치는 AMHS 트랙 루프를 따라 이동(코너 포함), 색/경고는 모델 예측.
- 정적 대시보드(static/index.html)를 서빙, /ws 로 상태 push.

실행:  cd src && uvicorn realtime_server:app --reload
브라우저:  http://127.0.0.1:8000
"""
import os
import io
import csv
import math
import time
import asyncio
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import fab_layout as FL
from telemetry_store import TelemetryStore
from dataset_quality import RoboticsDataQualityMonitor, records_from_agv_snapshot
from drift_monitor import DataDriftMonitor

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "..", "data", "processed")
STATIC = os.path.join(_HERE, "static")
KOR = {"E-ENV-C": "교차로 충돌위험", "E-ENV-O": "경로 장애물", "E-INF-A": "도어 연동",
       "E-INF-E": "층간리프트 연동", "E-RBT-B": "배터리 저하", "E-RBT-E": "긴급정지",
       "E-RBT-N": "통신 끊김", "E-RBT-S": "센서 이상", "정상": "정상"}

# 진단(오류코드) → AI 판단 근거(주요 이상 신호). explain.py의 물리 신호 그룹 어휘와 정합.
CAUSE = {
    "E-RBT-B": ["배터리 잔량 추이 ↓", "누적 배터리 소모 ↑", "구동부 온도 ↑"],
    "E-RBT-S": ["센서 신호 이상 ↑", "이동 속도·진동 ↑", "진행각 동역학 불안정"],
    "E-RBT-E": ["긴급정지 신호 ↑", "이동 속도 급감", "구동부 온도 ↑"],
    "E-RBT-N": ["통신 오프라인 여부 ↑", "상태 보고 지연"],
    "E-ENV-C": ["충돌 신호 ↑", "주변 혼잡도 ↑", "이동 속도·진동 ↑"],
    "E-ENV-O": ["장애물 감지 ↑", "경로 이탈", "이동 속도 급변"],
    "E-INF-A": ["도어 연동 대기", "정지 시간 ↑"],
    "E-INF-E": ["층간리프트 연동 대기", "정지 시간 ↑"],
    "정상": ["모든 신호 정상 범위"],
}

# 경고 등급(주의/경고/위험) — 진단 신뢰도(conf) 기준의 3단계 트리아지.
# 관제 화면에서 다수 경고를 한눈에 우선순위화(색·집계)하기 위한 계약값.
LEVELS = ("주의", "경고", "위험")


def alert_level(conf: float) -> str:
    """진단 신뢰도를 3단계 경고 등급으로 매핑. 높을수록 확실한 이상 → 우선 대응."""
    if conf >= 0.85:
        return "위험"
    if conf >= 0.60:
        return "경고"
    return "주의"


TREND_W = 12                                  # B3: 이력 타임라인 길이(최근 N틱)
_LVL_CODE = {"주의": 1, "경고": 2, "위험": 3}


def diag_code(pred: str, conf: float) -> int:
    """진단을 0~3 코드로. 0 정상 · 1 주의 · 2 경고 · 3 위험 (타임라인 셀 색상 기준)."""
    return 0 if pred == "정상" else _LVL_CODE[alert_level(conf)]


DRIFT_MARGIN = 0.5      # B2: 추세 방향 판정 임계(뒤 절반 평균 - 앞 절반 평균)


def trend_direction(trend: list) -> str:
    """진단 추세(0~3 코드열)의 방향 판정 — 드리프트 조기 감지.
    뒷부분 평균이 앞부분보다 임계 이상 높으면 '악화'(아직 정상이어도 나빠지는 중), 낮으면 '개선', 그 외 '안정'.
    → 고장이 터지기 '전에' 악화 추세를 잡아내는 예지보전의 핵심 지표."""
    if len(trend) < 4:
        return "안정"
    h = len(trend) // 2
    early = sum(trend[:h]) / h
    late = sum(trend[h:]) / (len(trend) - h)
    if late - early >= DRIFT_MARGIN:
        return "악화"
    if early - late >= DRIFT_MARGIN:
        return "개선"
    return "안정"


def health_index(trend: list, conf: float, warn: bool) -> int:
    """설비 건전도 지표(0~100). 순간 분류를 넘어 '최근 추세'를 종합한 자산 건전도 점수.
    - 최근 진단 추세(0~3 코드)의 평균 심각도가 높을수록 감점(간헐 이상 이력 반영)
    - 현재 이상이면 신뢰도에 비례해 추가 감점
    100=정상·안정, 낮을수록 정비 우선순위 ↑ → 예지보전 Health Index 컨셉."""
    if not trend:
        return 100
    mean_sev = sum(trend) / len(trend)              # 0~3
    score = 100 - mean_sev * 22                      # 추세 심각도 페널티
    if warn:
        score -= 10 + conf * 18                      # 현재 이상 가중(신뢰도 비례)
    return int(max(2, min(100, round(score))))


def maint_advice(health: int, trend_dir: str) -> str:
    """건전도·추세 방향 → 정비 트리아지 권고(관제자가 바로 행동할 수 있는 한 줄)."""
    if health >= 80:
        return "정상 가동" if trend_dir != "악화" else "관찰(악화 추세)"
    if health >= 55:
        return "점검 권장(악화 추세)" if trend_dir == "악화" else "모니터링"
    if health >= 30:
        return "정비 권장"
    return "정비 필요 · 우선 대응"


def agv_sensors(i: int, n: int, pred: str) -> dict:
    """AGV 실시간 텔레메트리(진동·배터리·온도)를 결정론적으로 산출.
    replay 인덱스(i)에 위상을 고정해 재현 가능하고, 진단(pred)에 물리적으로 커플링한다.
    - 진동 vib(mm/s): 구동부 마모/충돌 → 상승
    - 배터리 batt(%): 방전 추이, 배터리 저하 시 저수준
    - 온도 temp(°C): 긴급정지/배터리 이상 시 상승
    """
    ph = i / max(n - 1, 1)                                   # 0~1 진행도
    wob = math.sin(i * 0.7) * 0.5 + math.sin(i * 0.23) * 0.3  # 결정론적 미세 변동
    vib = 2.0 + 0.6 * wob                                    # 기본 진동
    batt = 88.0 - 40.0 * ph                                  # 서서히 방전
    temp = 37.0 + 4.0 * (0.5 + 0.5 * math.sin(i * 0.11))     # 구동부 온도
    if pred == "E-RBT-B":
        batt = 24.0 - 8.0 * (0.5 + 0.5 * math.sin(i * 0.11)); temp += 9
    elif pred == "E-RBT-S":
        vib = 7.6 + 1.6 * wob
    elif pred == "E-RBT-E":
        vib = 5.4 + 1.2 * wob; temp += 16
    elif pred in ("E-ENV-C", "E-ENV-O"):
        vib = 6.4 + 1.4 * wob
    elif pred == "E-RBT-N":
        temp += 4
    return {"vib": round(vib, 2), "batt": round(max(batt, 4.0), 1), "temp": round(temp, 1)}

app = FastAPI(title="FAB AMHS 실시간 관제 서버")

rep = pd.read_parquet(f"{DATA}/replay.parquet")
robots = sorted(rep["robot"].unique())
TRAJ = {}
for rid in robots:
    g = rep[rep.robot == rid].sort_values("seq").reset_index(drop=True)
    px, py = g.px.to_numpy(), g.py.to_numpy()
    step = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
    TRAJ[rid] = dict(prog=step / (step[-1] + 1e-9), pred=g["pred"].to_numpy(),
                     conf=g["conf"].to_numpy(), n=len(g))
PLAN = FL.build_agv_plan(robots)
LAYOUT = FL.build_layout()
P = {"v": 0.0}     # 전역 진행도(0~1 순환)
# 시계열 데이터 계층(진단 이벤트 적재/집계/조회). 기본 인메모리, TELEMETRY_DB로 파일 durable.
STORE = TelemetryStore(os.environ.get("TELEMETRY_DB", ":memory:"))
QUALITY = RoboticsDataQualityMonitor()
DRIFT = DataDriftMonitor()


def snapshot():
    p = P["v"]; agvs = []; per = [0, 0, 0]; alerts = []
    for a in PLAN:
        t = TRAJ[a["robot"]]; n = t["n"]
        i = int(((p + a.get("toff", 0.0)) % 1.0) * (n - 1))    # AGV별 시점 오프셋 → 고장 분산
        s = (t["prog"][i] + a["phase"]) % 1.0
        x, y, ang = a["route"].at(s)
        pred = t["pred"][i]; warn = pred != "정상"
        per[a["floor"]] += int(warn)
        conf = round(float(t["conf"][i]), 3)
        level = alert_level(conf) if warn else None
        lo = max(0, i - (TREND_W - 1))         # B3: 최근 N틱 진단 추세(0~3 코드)
        trend = [diag_code(t["pred"][k], round(float(t["conf"][k]), 3)) for k in range(lo, i + 1)]
        trend_dir = trend_direction(trend)      # B2: 악화/개선/안정 방향(드리프트 조기 감지)
        health = health_index(trend, conf, warn)         # B4: 자산 건전도 지표(추세 종합)
        advice = maint_advice(health, trend_dir)         # B4: 정비 트리아지 권고
        item = {"id": a["id"], "x": round(x, 2), "y": round(y, 2), "ang": round(ang, 1),
                "floor": a["floor"], "status": "warn" if warn else "ok",
                "pred": pred, "label": KOR.get(pred, pred), "conf": conf, "level": level,
                "sensors": agv_sensors(i, n, pred), "cause": CAUSE.get(pred, CAUSE["정상"]),
                "trend": trend, "trend_dir": trend_dir, "health": health, "advice": advice}
        agvs.append(item)
        if warn:
            alerts.append({"id": a["id"], "label": item["label"], "conf": conf,
                           "floor": a["floor"], "level": level})
    w = sum(per)
    by_level = {L: sum(al["level"] == L for al in alerts) for L in LEVELS}
    deteriorating = sum(a["trend_dir"] == "악화" for a in agvs)   # B2: 악화 추세(드리프트) 대수
    maint_due = sum(a["health"] < 55 for a in agvs)              # B4: 정비 필요(건전도<55) 대수
    avg_health = round(sum(a["health"] for a in agvs) / max(len(agvs), 1))
    return {"type": "state", "p": round(p, 4), "agvs": agvs,
            "kpi": {"total": len(PLAN), "ok": len(PLAN) - w, "warn": w,
                    "per_floor": per, "by_level": by_level, "deteriorating": deteriorating,
                    "maint_due": maint_due, "avg_health": avg_health},
            "alerts": alerts}


@app.on_event("startup")
async def _advance():
    async def loop():
        tick = 0
        while True:
            P["v"] = (P["v"] + 0.0035) % 1.0
            tick += 1
            if tick % 10 == 0:                         # ~2Hz로 진단 이벤트 시계열 적재
                STORE.record(time.time(), snapshot()["agvs"])
                if tick % 200 == 0:                    # 주기적 보존정책(오래된 이벤트 정리)
                    STORE.prune()
            await asyncio.sleep(0.05)
    asyncio.create_task(loop())


@app.get("/api/layout")
def api_layout():
    return JSONResponse(LAYOUT)


@app.get("/api/snapshot")
def api_snapshot():
    return JSONResponse(snapshot())


_HIST_COLS = ["ts", "pred", "conf", "level", "health", "vib", "batt", "temp"]


@app.get("/api/history")
def api_history(agv: str, limit: int = 200, fmt: str = "json"):
    """설비 1대의 진단 이벤트 시계열 이력(최신순). fmt=csv 로 리포팅용 CSV 반출."""
    rows = STORE.history(agv, min(max(limit, 1), 1000))
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=_HIST_COLS)
        w.writeheader()
        w.writerows(rows)
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename={agv}_events.csv"})
    return JSONResponse(rows)


@app.get("/api/stats")
def api_stats():
    """세션 누적 진단 이벤트 롤업 집계(총계·등급별·층별·최다결함·평균건전도)."""
    return JSONResponse(STORE.stats())


@app.get("/api/trend")
def api_trend(bucket: int = 60, n: int = 15):
    """시간 버킷 롤업(다운샘플링) — 버킷별 이벤트·평균건전도·등급 집계. 관제 추이 차트 소스."""
    return JSONResponse(STORE.trend(min(max(bucket, 10), 3600), min(max(n, 2), 120)))


@app.get("/api/reliability")
def api_reliability(agv: str = None):
    """신뢰성 지표(MTBF·MTTR·가용도) — 설비 1대(agv=) 또는 플릿 전체(worst 5 포함)."""
    return JSONResponse(STORE.reliability(agv=agv, n_total=len(PLAN)))


@app.get("/api/data-quality")
def api_data_quality():
    """로보틱스 학습 데이터 QA 지표.

    현재 AGV 스냅샷을 표준 학습 데이터 레코드로 변환해 스키마 정합성, annotation coverage,
    QA pass rate, ingest success rate, rework rate를 계산한다. 제조 로보틱스 데이터셋 구축에서
    필요한 데이터 거버넌스 지표의 작은 프로토타입이다.
    """
    now = time.time()
    records = records_from_agv_snapshot(snapshot()["agvs"], now)
    return JSONResponse(QUALITY.evaluate(records).as_dict())


@app.get("/api/drift")
def api_drift():
    """실시간 입력 분포 드리프트 감지.

    현재 AGV 텔레메트리(vibration/battery/temperature/health/confidence)와
    경고율을 기준 운전 프로파일과 비교해, 학습·검증 때 기대한 operating envelope에서
    벗어나는지 feature-level z-score로 감시한다.
    """
    return JSONResponse(DRIFT.evaluate(snapshot()["agvs"]))


@app.get("/metrics")
def metrics():
    """Prometheus 텍스트 포맷 메트릭 — 운영 모니터링(Grafana 등) 표준 연동점.
    관제 KPI·데이터 계층·신뢰성 지표를 게이지로 노출한다."""
    snap = snapshot()
    s = snap["kpi"]
    st = STORE.stats()
    rel = STORE.reliability(n_total=len(PLAN))
    drift = DRIFT.evaluate(snap["agvs"])
    g = lambda name, help_, val: (f"# HELP {name} {help_}\n# TYPE {name} gauge\n{name} {val}")
    lines = [
        g("fab_agv_total", "가동 AGV 수", s["total"]),
        g("fab_agv_warn", "이상 감지 AGV 수", s["warn"]),
        g("fab_agv_deteriorating", "악화 추세(드리프트) AGV 수", s["deteriorating"]),
        g("fab_agv_maintenance_due", "정비 필요(건전도<55) AGV 수", s["maint_due"]),
        g("fab_fleet_avg_health", "플릿 평균 건전도(0~100)", s["avg_health"]),
        g("fab_events_stored", "시계열 스토어 적재 이벤트 수", st["total"]),
        g("fab_data_qa_pass_rate", "로보틱스 데이터 QA 통과율(0~1)",
          QUALITY.evaluate(records_from_agv_snapshot(snap["agvs"], time.time())).as_dict()["qa_pass_rate"]),
        g("fab_data_drift_score", "실시간 입력 분포 드리프트 점수(max absolute z-score)", drift["score"]),
        g("fab_data_drift_features", "드리프트 상태인 feature 수", len(drift["drifted_features"])),
        g("fab_data_drift_fault_rate", "현재 스냅샷 이상 AGV 비율", drift["fault_rate"]["current"]),
        g("fab_fleet_availability", "플릿 가용도(0~1)", rel["availability"]),
        g("fab_fleet_mttr_seconds", "평균 복구 시간(초)", rel["mttr"]),
        g("fab_fleet_mtbf_seconds", "평균 고장 간격(초)", rel["mtbf"] if rel["mtbf"] is not None else 0),
        g("fab_failure_episodes_total", "관측창 내 고장 에피소드 수", rel["episodes"]),
    ]
    return PlainTextResponse("\n".join(lines) + "\n",
                             media_type="text/plain; version=0.0.4; charset=utf-8")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            await sock.send_json(snapshot())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
    except Exception:
        return


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/twin")
def twin():
    """3D 디지털 트윈 관제(Three.js) — 태블릿 터치 조작 + 실시간 AI 진단."""
    return FileResponse(os.path.join(STATIC, "twin.html"))


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
