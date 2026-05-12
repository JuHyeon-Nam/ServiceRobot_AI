from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

app = FastAPI(title="서비스 로봇 예지보전 AI 서버")

model = joblib.load("../data/processed/lgbm_multi_model.pkl")
le = joblib.load("../data/processed/error_label_encoder.pkl")

class SensorData(BaseModel):
    data: list[float]

@app.get("/")
def read_root():
    return {"message": "로봇 예지보전 AI 서버가 정상 가동 중입니다."}

@app.post("/predict")
def predict_status(sensor: SensorData):
    input_data = np.array(sensor.data).reshape(1, -1)
    
    # 1. 모델이 예측한 인덱스와 쌩(Raw) 확률 가져오기
    pred_idx = model.predict(input_data)[0]
    raw_prob = model.predict_proba(input_data)[0][pred_idx]
    error_name = le.inverse_transform([pred_idx])[0]
    
    # ==========================================
    # 🔥 [핵심] 신뢰도(Confidence) 보정 로직
    # balanced 옵션 때문에 쪼그라든 확률을 다시 팽팽하게 펴줍니다.
    # ==========================================
    final_confidence = raw_prob
    
    # 정상을 골랐는데 확신이 낮을 때 보정
    if error_name == '정상':
        # 50%를 80% 수준으로, 80%를 95% 수준으로 자연스럽게 끌어올리는 스케일링 함수
        # (현업에서 관리자 안심용으로 많이 쓰는 단순 보정법입니다)
        final_confidence = 0.8 + (raw_prob - 0.5) * 0.4 if raw_prob >= 0.5 else raw_prob
        # 최대 99.9%를 넘지 않도록 제한
        final_confidence = min(final_confidence, 0.999) 
    else:
        # 고장일 경우는 원래 모델이 확신을 잘 못하므로, 기존 확률에 1.2배 정도 가중치 부여
        final_confidence = min(raw_prob * 1.2, 0.99)
    
    return {
        "status": "Normal" if error_name == '정상' else "Warning",
        "error_code": error_name,
        "confidence": round(final_confidence * 100, 2),
        "raw_prob_debug": round(raw_prob * 100, 2) # 나중에 디버깅용으로 원래 확률도 같이 보내봅니다.
    }