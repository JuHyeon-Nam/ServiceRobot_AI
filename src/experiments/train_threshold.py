import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_threshold_model():
    print("데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    if len(X.shape) == 3:
        print(f"🔄 3차원 데이터를 2차원으로 변환: {X.shape} -> ", end="")
        X = X.reshape(X.shape[0], -1)
        print(f"변경 완료 {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🤖 뼈대가 될 LightGBM 모델 학습 시작... (Optuna 최적화 세팅 차용)")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.015,
        num_leaves=212,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("\n검증 데이터 예측(확률 추출) 중... 📊")
    # 1. 단순 정답이 아니라, 0~9번일 '확률'을 전부 뽑아냅니다. (Shape: 데이터수 x 10)
    probs = model.predict_proba(X_val)

    # 2. 기본 예측 (가장 확률이 높은 것)
    y_pred = np.argmax(probs, axis=1)

    # ==========================================
    # 🔥 Threshold Moving (임계값 조정) 작전 🔥
    # ==========================================
    # 6번, 7번 고장일 확률이 단 5% (0.05)만 넘어도 무조건 6, 7번으로 강제 판정!
    THRESHOLD_6 = 0.05 
    THRESHOLD_7 = 0.05

    print(f"🚨 규칙 적용: 6번, 7번 고장 확률이 {THRESHOLD_6*100}%만 넘어도 강제 경고 발송!")
    
    for i in range(len(probs)):
        if probs[i, 6] > THRESHOLD_6:
            y_pred[i] = 6
        elif probs[i, 7] > THRESHOLD_7:
            y_pred[i] = 7

    acc = accuracy_score(y_val, y_pred)
    print("\n========================================")
    print(f"🎯 Threshold Moving 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

    with open('../data/processed/threshold_lgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장 완료! (data/processed/threshold_lgbm_model.pkl)")

if __name__ == "__main__":
    train_threshold_model()