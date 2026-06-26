"""
realtime_server.py — 실시간 AGV 관제 서버 (FastAPI + WebSocket)
- 학습된 PdM 모델의 진단을 AGV 플릿 상태로 실시간 스트리밍.
- AGV 위치는 AMHS 트랙 루프를 따라 이동(코너 포함), 색/경고는 모델 예측.
- 정적 대시보드(static/index.html)를 서빙, /ws 로 상태 push.

실행:  cd src && uvicorn realtime_server:app --reload
브라우저:  http://127.0.0.1:8000
"""
import os
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
        item = {"id": a["id"], "x": round(x, 2), "y": round(y, 2), "ang": round(ang, 1),
                "floor": a["floor"], "status": "warn" if warn else "ok",
                "pred": pred, "label": KOR.get(pred, pred), "conf": round(float(t["conf"][i]), 3)}
        agvs.append(item)
        if warn:
            alerts.append({"id": a["id"], "label": item["label"], "conf": item["conf"], "floor": a["floor"]})
    w = sum(per)
    return {"type": "state", "p": round(p, 4), "agvs": agvs,
            "kpi": {"total": len(PLAN), "ok": len(PLAN) - w, "warn": w, "per_floor": per},
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


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
