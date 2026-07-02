"""
realtime_server.py — 실시간 AGV 관제 서버 (FastAPI + WebSocket)
- 학습된 PdM 모델의 진단을 AGV 플릿 상태로 실시간 스트리밍.
- AGV 위치는 AMHS 트랙 루프를 따라 이동(코너 포함), 색/경고는 모델 예측.
- 정적 대시보드(static/index.html)를 서빙, /ws 로 상태 push.

실행:  cd src && uvicorn realtime_server:app --reload
브라우저:  http://127.0.0.1:8000
"""
import os
import math
import asyncio
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import fab_layout as FL

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


def snapshot():
    p = P["v"]; agvs = []; per = [0, 0, 0]; alerts = []
    for a in PLAN:
        t = TRAJ[a["robot"]]; n = t["n"]; i = int(p * (n - 1))
        s = (t["prog"][i] + a["phase"]) % 1.0
        x, y, ang = a["route"].at(s)
        pred = t["pred"][i]; warn = pred != "정상"
        per[a["floor"]] += int(warn)
        conf = round(float(t["conf"][i]), 3)
        level = alert_level(conf) if warn else None
        item = {"id": a["id"], "x": round(x, 2), "y": round(y, 2), "ang": round(ang, 1),
                "floor": a["floor"], "status": "warn" if warn else "ok",
                "pred": pred, "label": KOR.get(pred, pred), "conf": conf, "level": level,
                "sensors": agv_sensors(i, n, pred), "cause": CAUSE.get(pred, CAUSE["정상"])}
        agvs.append(item)
        if warn:
            alerts.append({"id": a["id"], "label": item["label"], "conf": conf,
                           "floor": a["floor"], "level": level})
    w = sum(per)
    by_level = {L: sum(al["level"] == L for al in alerts) for L in LEVELS}
    return {"type": "state", "p": round(p, 4), "agvs": agvs,
            "kpi": {"total": len(PLAN), "ok": len(PLAN) - w, "warn": w,
                    "per_floor": per, "by_level": by_level},
            "alerts": alerts}


@app.on_event("startup")
async def _advance():
    async def loop():
        while True:
            P["v"] = (P["v"] + 0.0035) % 1.0
            await asyncio.sleep(0.05)
    asyncio.create_task(loop())


@app.get("/api/layout")
def api_layout():
    return JSONResponse(LAYOUT)


@app.get("/api/snapshot")
def api_snapshot():
    return JSONResponse(snapshot())


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
