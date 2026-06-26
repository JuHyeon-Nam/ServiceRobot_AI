import numpy as np
import joblib
import pandas as pd

def test_lgbm_inference():
    # 1. 모델 및 번역기 로드
    model = joblib.load("../data/processed/lgbm_multi_model.pkl")
    le = joblib.load("../data/processed/error_label_encoder.pkl")
    
    # 2. 테스트용 데이터 로드 (X_train 중 일부 사용)
    X_test_all = np.load("../data/processed/X_train.npy")
    
    # 3. 임의의 상황 선정 (예: 500번째 데이터)
    sample_idx = 500
    sample_input = X_test_all[sample_idx] # (30, 7) 형태
    
    # 4. 데이터 형태 변환 (LGBM용으로 납작하게)
    input_flat = sample_input.reshape(1, -1)
    
    # 5. 예측
    pred_idx = model.predict(input_flat)[0]
    pred_prob = model.predict_proba(input_flat)[0][pred_idx]
    error_name = le.inverse_transform([pred_idx])[0]
    
    print("="*40)
    print("🤖 실시간 로봇 상태 진단 결과")
    print("-"*40)
    if error_name == '정상':
        print(f"✅ 상태: 정상 작동 중")
    else:
        print(f"🚨 경고: 고장 징후 감지!")
        print(f"🔎 원인: {error_name}")
    print(f"📊 신뢰도: {pred_prob * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    test_lgbm_inference()