import numpy as np
import pickle
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_ultimate_fft_features(X_raw):
    N, seq_len, num_sensors = X_raw.shape
    X_flat = X_raw.reshape(N, -1)
    time_mean = np.mean(X_raw, axis=1)
    time_std = np.std(X_raw, axis=1)
    time_diff = np.mean(np.diff(X_raw, axis=1), axis=1)
    
    fft_features = np.abs(np.fft.fft(X_raw, axis=1))
    fft_half = fft_features[:, :15, :]
    X_fft_flat = fft_half.reshape(N, -1)
    
    return np.hstack([X_flat, time_mean, time_std, time_diff, X_fft_flat])

def train_hybrid_system():
    print("최종 하이브리드 알고리즘 데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 30, 7)

    X_ultimate = create_ultimate_fft_features(X)
    X_train, X_val, y_train, y_val = train_test_split(X_ultimate, y, test_size=0.2, random_state=42, stratify=y)

    # 1. 뼈대 모델: 아까 95.8%를 찍었던 최고 성능의 LightGBM
    print("\n🤖 [Stage 1] 최강 FFT 기반 LightGBM 훈련 시작...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.02, num_leaves=256,
        min_child_samples=5, class_weight='balanced', random_state=42, n_jobs=-1
    )
    lgbm_model.fit(X_train, y_train)

    # 2. 🛡️ 희귀 고장 저격수: Isolation Forest (비정상 탐지)
    print("\n🌲 [Stage 2] 희귀 고장을 고립시키는 Isolation Forest 훈련 중...")
    # 정상 데이터(9번)와 다수 고장 데이터를 기준으로 '정상적인 흐름'을 학습
    # contamination=0.005 : 전체 데이터 중 약 0.5%를 '완전 이상한 놈(6, 7번 등)'으로 강제 분류하라는 지시
    iso_forest = IsolationForest(n_estimators=300, contamination=0.005, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)

    print("\n⚔️ [Stage 3] 하이브리드 검증 및 6번 강제 포획 진행 중... 📊")
    # LightGBM의 기본 예측
    y_pred_lgbm = lgbm_model.predict(X_val)
    
    # Isolation Forest의 이상치 탐지 (-1 이면 이상치, 1 이면 정상)
    anomaly_preds = iso_forest.predict(X_val)
    
    # 🔥 하이브리드 결합 룰:
    # 만약 LightGBM이 '정상(9)'이라고 했거나, 헷갈려 하는데
    # Isolation Forest가 "이거 구조적으로 완전 이상한 놈(-1)이야!" 라고 판정했다면
    # 이건 평범한 고장이 아니라 '6번(희귀 고장)'일 확률이 매우 높다고 강제 치환합니다.
    for i in range(len(y_pred_lgbm)):
        if anomaly_preds[i] == -1 and y_pred_lgbm[i] == 9:
            y_pred_lgbm[i] = 6  

    acc = accuracy_score(y_val, y_pred_lgbm)
    print("\n========================================")
    print(f"🌟 최종 하이브리드(LGBM + IsoForest) 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred_lgbm))

if __name__ == "__main__":
    train_hybrid_system()