import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_time_features(X_raw):
    # X_raw shape: (N, 30, 7)
    N, seq_len, num_sensors = X_raw.shape
    
    # 기존 2차원으로 편 데이터 (기본 베이스)
    X_flat = X_raw.reshape(N, -1)
    
    # 🚀 시계열 통계 피처 생성 (LightGBM에게 시간 흐름을 가르치는 무기)
    print("🧠 시계열 통계 피처 생성 중 (이동평균, 표준편차, 변화율)...")
    
    # 1. 30시점 전체의 평균 (N, 7)
    time_mean = np.mean(X_raw, axis=1)
    # 2. 30시점 전체의 표준편차 - 진동 감지 (N, 7)
    time_std = np.std(X_raw, axis=1)
    # 3. 직전 시점과의 차이(변화량)의 평균 (N, 7)
    time_diff = np.mean(np.diff(X_raw, axis=1), axis=1)
    
    # 모든 피처를 옆으로 결합합니다.
    X_advanced = np.hstack([X_flat, time_mean, time_std, time_diff])
    return X_advanced

def train_advanced_model():
    print("데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    # 원본 3차원 형태로 강제 복원 (시계열 연산을 위해)
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 30, 7)

    # 🔥 통계학적 시계열 피처 엔지니어링 적용!
    X_engineered = create_time_features(X)
    print(f"📈 피처 고도화 완료! (기존 피처 수: 210개 -> 고도화 후: {X_engineered.shape[1]}개)")

    # 학습/검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🤖 고도화된 LightGBM 모델 훈련 시작... (이번엔 다를 겁니다)")
    # 6, 7번 미세 패턴을 낚아채도록 num_leaves를 크게 잡고 min_child_samples를 낮춥니다.
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=10, 
        class_weight='balanced', # 황금 밸런스 유지
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("\n검증 데이터 예측 및 최종 성능 평가 중... 📊")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print("\n========================================")
    print(f"🏆 피처 엔지니어링 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

    # 최고 성능 모델 저장
    with open('../data/processed/optuna_lgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 최고 점수 갱신 모델 저장 완료!")

if __name__ == "__main__":
    train_advanced_model()