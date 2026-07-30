"""
서비스 로봇 실시간 예지보전(PdM) AI API 서버
- 강화 모델(robot_pdm_enhanced.txt, 2.8MB)을 CPU로 추론.
- 입력: 30시점 동적센서 7종 윈도우 + 정적/맥락 컨텍스트.
- 동적센서는 시퀀스 피처(flatten/mean/std/trend/FFT)로, 정적은 그대로 부착.
"""
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from explain import explain_prediction
from model_card import build_model_card
from pdm_runtime import load_runtime, make_features as runtime_make_features

DATA = "../data/processed"

booster, mmeta, emeta = load_runtime(DATA)
MODEL_CARD = build_model_card(DATA)

CLASSES = mmeta["classes"]        # 모델 확률열 순서(=정렬된 클래스 인덱스)
NAMES = mmeta["class_names"]       # 실제 errorCode 문자열
DEVMAP = emeta["devtype_map"]
MAINMAP = emeta.get("mainstate_map", {})
CROWDMAP = emeta.get("crowd_map", {"LOW": 0, "MIDDLE": 1, "HIGH": 2})

# 실제 errorCode 접두 기반 카테고리(임의 명칭 만들지 않음)
CATEGORY = {"E-ENV": "환경 이상", "E-INF": "인프라 이상", "E-RBT": "로봇 본체 이상"}

app = FastAPI(title="서비스 로봇 실시간 예지보전 AI 서버 🤖")
print(f"✅ 강화 모델 로드 (피처 {mmeta['n_features']}개, 공식 Validation acc {mmeta['val_acc']*100:.2f}%)")


def make_features(window, ctx):
    return runtime_make_features(window, ctx, emeta)


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


@app.get("/model-card")
def model_card():
    return MODEL_CARD


@app.post("/predict")
def predict(payload: PredictIn):
    w = payload.window
    if len(w) != 30 or any(len(r) != 7 for r in w):
        return {"error": "window는 30x7 형태여야 합니다 (시점30 x 센서7)."}
    feats = make_features(w, payload.context)
    probs = booster.predict(feats)[0]
    top = int(np.argmax(probs))
    code = NAMES[top]
    cat = "정상" if code == "정상" else CATEGORY.get(code[:5], "이상")
    # 진단 근거(설명가능성): 어떤 물리 신호가 이 진단을 이끌었는가 Top3
    reason = explain_prediction(booster, feats, top, n_classes=len(CLASSES),
                                n_features=mmeta["n_features"], top_k=3)
    return {
        "error_code": code,
        "category": cat,
        "confidence": f"{probs[top]*100:.2f}%",
        "action_required": "None" if code == "정상" else "Immediate Inspection",
        "reason": reason,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
