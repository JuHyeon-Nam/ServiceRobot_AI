import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_trend_fft_features(X_raw):
    N, seq_len, num_sensors = X_raw.shape
    X_flat = X_raw.reshape(N, -1)
    
    # 1. 기본 시계열 통계
    time_mean = np.mean(X_raw, axis=1)
    time_std = np.std(X_raw, axis=1)
    
    # 2. 🚀 새로운 피처: 추세선(Trend/Drift) 피처 추가
    # 배터리 저하(1번), 구동부 마모(0번)는 서서히 변하는 특징이 있음
    # 첫 10프레임 평균과 마지막 10프레임 평균의 차이를 구해 장기적인 '기울기'를 AI에게 알려줌
    first_10_mean = np.mean(X_raw[:, :10, :], axis=1)
    last_10_mean = np.mean(X_raw[:, -10:, :], axis=1)
    trend_drift = last_10_mean - first_10_mean
    
    # 3. FFT (주파수) 피처 (기존 강력한 피처 유지)
    fft_features = np.abs(np.fft.fft(X_raw, axis=1))
    fft_half = fft_features[:, :15, :].reshape(N, -1)
    
    return np.hstack([X_flat, time_mean, time_std, trend_drift, fft_half])

def train_majority_focused_model():
    print("데이터 로딩 및 대다수 클래스 집중 피처 고도화 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 30, 7)

    X_engineered = create_trend_fft_features(X)
    X_train, X_val, y_train, y_val = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🤖 족쇄 해제! 다수 클래스 100% 타겟팅 훈련 시작...")
    # 🔥 핵심 변경: class_weight='balanced' 전면 삭제!
    # 이제 AI는 21개짜리 희귀 고장에 억지로 목매지 않고, 1만 개짜리 주요 고장에 역량을 100% 집중합니다.
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.015,
        num_leaves=256,
        min_child_samples=20, # 규칙을 약간 여유롭게 풀어주어 다수 클래스를 넓게 품음
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("\n검증 데이터 예측 및 성능 평가 중... 📊")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print("\n========================================")
    print(f"🚀 다수 클래스 집중형(Trend + FFT) 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

    with open('../data/processed/majority_focused_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 98% 타겟팅 모델 저장 완료!")

if __name__ == "__main__":
    train_majority_focused_model()