"""
서비스 로봇 실시간 예지보전(PdM) AI API 서버
- 휴대용 LightGBM 모델(robot_pdm_portable.txt, 5.5MB)을 CPU로 추론.
- 입력: 30시점 x 7센서 = 210개 raw 값. 서버가 학습과 동일한 피처(336개)로 변환.
"""
import json
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="서비스 로봇 실시간 예지보전 AI 서버 🤖")

MODEL_PATH = "../data/processed/robot_pdm_portable.txt"
META_PATH = "../data/processed/robot_pdm_meta.json"

booster = lgb.Booster(model_file=MODEL_PATH)
meta = json.load(open(META_PATH, encoding="utf-8"))
CLASSES = meta["classes"]  # 모델이 내놓는 확률 열 순서 -> 실제 클래스 번호
print(f"✅ 휴대용 모델 로드 완료 (피처 {meta['n_features']}개, holdout acc {meta['holdout_acc']*100:.2f}%)")

STATUS_MAP = {
    0: "E-RBT-A (구동부 이상)", 1: "E-RBT-B (배터리 전압 저하)",
    2: "E-RBT-C (통신 모듈 과열)", 3: "E-RBT-D (센서 오동작)",
    4: "E-RBT-E (바퀴 슬립 감지)", 5: "E-RBT-F (범퍼 충격 감지)",
    6: "E-RBT-G (모터 전류 급증 - 희귀 고장)", 7: "E-RBT-H (라이다 차단 신호)",
    8: "E-RBT-I (엔코더 오차 초과)", 9: "Normal (정상 작동 중)",
}


def create_features(X_raw):
    """train_portable.py와 100% 동일해야 함. (N,30,7) -> (N,336)"""
    N = X_raw.shape[0]
    X_flat = X_raw.reshape(N, -1)
    t_mean = np.mean(X_raw, axis=1)
    t_std = np.std(X_raw, axis=1)
    trend = np.mean(X_raw[:, -10:, :], axis=1) - np.mean(X_raw[:, :10, :], axis=1)
    fft_half = np.abs(np.fft.rfft(X_raw, axis=1))[:, :15, :].reshape(N, -1)
    return np.hstack([X_flat, t_mean, t_std, trend, fft_half]).astype(np.float32)


class RobotSensorInput(BaseModel):
    sensor_data: list  # 210개 (30시점 x 7센서)


@app.get("/")
def root():
    return {"message": "로봇 예지보전 AI 서버 정상 가동 중", "model_acc": meta["holdout_acc"]}


@app.post("/predict")
def predict_robot_status(payload: RobotSensorInput):
    raw = np.asarray(payload.sensor_data, dtype=np.float64).reshape(30, 7)
    feats = create_features(raw[np.newaxis, ...])  # (1, 336)

    probs = booster.predict(feats)[0]              # 클래스 확률 (CLASSES 순서)
    top = int(np.argmax(probs))
    pred_class = CLASSES[top]
    confidence = float(probs[top]) * 100

    return {
        "status_code": pred_class,
        "status_display": STATUS_MAP[pred_class],
        "confidence": f"{confidence:.2f}%",
        "action_required": "None" if pred_class == 9 else "Immediate Inspection",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
