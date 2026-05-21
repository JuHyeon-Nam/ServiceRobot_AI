import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_ultimate_fft_features(X_raw):
    # 1. 기본 형태 (N, 30, 7)
    N, seq_len, num_sensors = X_raw.shape
    X_flat = X_raw.reshape(N, -1)
    
    # 2. 통계 피처 (아까 95.6%를 달성했던 뼈대)
    time_mean = np.mean(X_raw, axis=1)
    time_std = np.std(X_raw, axis=1)
    time_diff = np.mean(np.diff(X_raw, axis=1), axis=1)
    
    # 3. 🚀 고속 푸리에 변환 (FFT) - 주파수 대역의 숨은 단서 추출!
    print("🌊 센서 데이터의 주파수(FFT) 성분을 엑스레이처럼 추출합니다...")
    # 시간 도메인을 주파수 도메인으로 변환 후 복소수 크기(절대값)만 사용
    fft_features = np.abs(np.fft.fft(X_raw, axis=1))
    
    # 나이퀴스트 이론에 따라 절반의 주파수 대역(15개)만 사용 (나머지는 거울처럼 대칭임)
    fft_half = fft_features[:, :15, :]
    X_fft_flat = fft_half.reshape(N, -1)
    
    # 4. [기존 평면 데이터 + 통계 흐름 + 주파수 대역] 3단 콤보 결합
    return np.hstack([X_flat, time_mean, time_std, time_diff, X_fft_flat])

def train_sota_fft_model():
    print("최종 SOTA 알고리즘 데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    # 원본 3차원으로 강제 복원
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 30, 7)

    # 영혼을 끌어모은 피처 엔지니어링 실행
    X_ultimate = create_ultimate_fft_features(X)
    print(f"📈 주파수(FFT) 피처 결합 완료! (최종 피처 수: {X_ultimate.shape[1]}개)")
    
    X_train, X_val, y_train, y_val = train_test_split(X_ultimate, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🤖 최강 FFT 기반 LightGBM 훈련 시작...")
    # 주파수의 미세한 차이를 낚아채기 위해 잎사귀 제한(min_child_samples)을 5로 최소화
    model = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=256,
        min_child_samples=5, 
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("\n검증 데이터 예측 및 성능 평가 중... 📊")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print("\n========================================")
    print(f"🚀 SOTA (FFT + Time + LGBM) 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

    with open('../data/processed/sota_fft_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 98% 돌파를 위한 마스터 모델 저장 완료!")

if __name__ == "__main__":
    train_sota_fft_model()