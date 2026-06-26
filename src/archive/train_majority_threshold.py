import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_trend_fft_features(X_raw):
    N, seq_len, num_sensors = X_raw.shape
    X_flat = X_raw.reshape(N, -1)
    
    time_mean = np.mean(X_raw, axis=1)
    time_std = np.std(X_raw, axis=1)
    
    first_10_mean = np.mean(X_raw[:, :10, :], axis=1)
    last_10_mean = np.mean(X_raw[:, -10:, :], axis=1)
    trend_drift = last_10_mean - first_10_mean
    
    fft_features = np.abs(np.fft.fft(X_raw, axis=1))
    fft_half = fft_features[:, :15, :].reshape(N, -1)
    
    return np.hstack([X_flat, time_mean, time_std, trend_drift, fft_half])

def optimize_majority_thresholds():
    print("데이터 로딩 및 최고 성능 모델 불러오는 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 30, 7)

    X_engineered = create_trend_fft_features(X)
    _, X_val, _, y_val = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)

    # 방금 저장한 최고 성능 95.9% 짜리 모델 로드
    try:
        with open('../data/processed/majority_focused_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("🚨 이전 모델 파일이 없습니다! 먼저 train_majority_focus.py를 실행해 주세요.")
        return

    print("\n🔮 0번, 1번 대다수 클래스 정답률 극대화 튜닝 중...")
    # 단순 정답이 아니라 '확률'을 전부 뽑아냅니다.
    probs = model.predict_proba(X_val)
    
    # 기본 예측 (가장 확률이 높은 것)
    y_pred = np.argmax(probs, axis=1)

    # 🔥 다수 클래스 임계값 조정 (소심함 타파!)
    # 0번과 1번의 확률이 35% 이상이면, 기존 1등(예: 9번)의 확률이 조금 더 높더라도 강제로 0, 1번으로 바꿔버림!
    THRESHOLD_0 = 0.35 
    THRESHOLD_1 = 0.35

    print(f"🚨 규칙 적용: 0번, 1번 확률이 {THRESHOLD_0*100}%만 넘어도 강제 정답 처리!")
    
    for i in range(len(probs)):
        # 만약 이미 정답이 0이나 1이면 건드리지 않음
        if y_pred[i] in [0, 1]:
            continue
            
        # 0번일 확률이 35%를 넘고, 1번 확률보다 높으면 0번으로 픽스
        if probs[i, 0] >= THRESHOLD_0 and probs[i, 0] >= probs[i, 1]:
            y_pred[i] = 0
        # 1번일 확률이 35%를 넘고, 0번 확률보다 높으면 1번으로 픽스
        elif probs[i, 1] >= THRESHOLD_1 and probs[i, 1] > probs[i, 0]:
            y_pred[i] = 1

    acc = accuracy_score(y_val, y_pred)
    print("\n========================================")
    print(f"🚀 임계값 튜닝 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

if __name__ == "__main__":
    optimize_majority_thresholds()