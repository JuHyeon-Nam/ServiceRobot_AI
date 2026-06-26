"""
서비스 로봇 실시간 예지보전(PdM) AI API 서버
- 강화 모델(robot_pdm_enhanced.txt, 2.8MB)을 CPU로 추론.
- 입력: 30시점 동적센서 7종 윈도우 + 정적/맥락 컨텍스트.
- 동적센서는 시퀀스 피처(flatten/mean/std/trend/FFT)로, 정적은 그대로 부착.
"""
import json
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

DATA = "../data/processed"
booster = lgb.Booster(model_file=f"{DATA}/robot_pdm_enhanced.txt")
mmeta = json.load(open(f"{DATA}/robot_pdm_enhanced_meta.json", encoding="utf-8"))
emeta = json.load(open(f"{DATA}/enhanced_meta.json", encoding="utf-8"))

CLASSES = mmeta["classes"]        # 모델 확률열 순서(=정렬된 클래스 인덱스)
NAMES = mmeta["class_names"]       # 실제 errorCode 문자열
DEVMAP = emeta["devtype_map"]
MAINMAP = emeta.get("mainstate_map", {})
CROWDMAP = emeta.get("crowd_map", {"LOW": 0, "MIDDLE": 1, "HIGH": 2})

# 실제 errorCode 접두 기반 카테고리(임의 명칭 만들지 않음)
CATEGORY = {"E-ENV": "환경 이상", "E-INF": "인프라 이상", "E-RBT": "로봇 본체 이상"}

app = FastAPI(title="서비스 로봇 실시간 예지보전 AI 서버 🤖")
print(f"✅ 강화 모델 로드 (피처 {mmeta['n_features']}개, 공식 Validation acc {mmeta['val_acc']*100:.2f}%)")


MODEL_DYN_IDX = [0, 1, 4, 5, 6]  # x,y 제외 (학습과 동일)

def make_features(window, ctx):
    Xf = np.asarray(window, dtype=np.float32).reshape(1, 30, 7)
    X = Xf[:, :, MODEL_DYN_IDX]
    eng = np.hstack([
        X.reshape(1, -1), np.mean(X, 1), np.std(X, 1),
        np.mean(X[:, -10:, :], 1) - np.mean(X[:, :10, :], 1),
        np.abs(np.fft.rfft(X, axis=1))[:, :15, :].reshape(1, -1),
    ])
    stat = np.array([[
        ctx.isOffline, ctx.nowCharging, ctx.emergencyStop, ctx.batteryUse,
        ctx.batteryCycleCount, ctx.distance, CROWDMAP.get(ctx.crowd, 1),
        DEVMAP.get(ctx.deviceType, 0), MAINMAP.get(ctx.mainState, 0),
    ]], dtype=np.float32)
    return np.hstack([eng, stat]).astype(np.float32)


class Context(BaseModel):
    deviceType: str = "안내로봇"
    mainState: str = "MOVE"
    crowd: str = "MIDDLE"
    isOffline: float = 0
    nowCharging: float = 0
    emergencyStop: float = 0
    batteryUse: float = 0
    batteryCycleCount: float = 0
    distance: float = 0


class PredictIn(BaseModel):
    window: list = Field(..., description="30x7 (시점 x [batteryLevel,speed,x,y,degree,collision,obstacle])")
    context: Context = Context()


@app.get("/")
def root():
    return {"status": "ok", "model_val_acc": mmeta["val_acc"], "classes": NAMES}


@app.get("/health")
def health():
    return {"ok": True, "n_features": mmeta["n_features"]}


@app.post("/predict")
def predict(payload: PredictIn):
    w = payload.window
    if len(w) != 30 or any(len(r) != 7 for r in w):
        return {"error": "window는 30x7 형태여야 합니다 (시점30 x 센서7)."}
    probs = booster.predict(make_features(w, payload.context))[0]
    top = int(np.argmax(probs))
    code = NAMES[top]
    cat = "정상" if code == "정상" else CATEGORY.get(code[:5], "이상")
    return {
        "error_code": code,
        "category": cat,
        "confidence": f"{probs[top]*100:.2f}%",
        "action_required": "None" if code == "정상" else "Immediate Inspection",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
