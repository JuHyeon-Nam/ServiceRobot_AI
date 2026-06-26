import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def train_precision_model():
    print("데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    if len(X.shape) == 3:
        print(f"🔄 3차원 데이터를 2차원으로 변환: {X.shape} -> ", end="")
        X = X.reshape(X.shape[0], -1)
        print(f"변경 완료 {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # 🎯 1. 스마트 타겟팅 SMOTE (핀셋 증식)
    # ==========================================
    print("\n✨ 스마트 타겟팅 SMOTE 진행 중... (6번, 7번 고장만 3,000개로 핀셋 증식!)")
    
    # 현재 클래스별 데이터 개수를 셉니다.
    counts = Counter(y_train)
    strategy = {k: v for k, v in counts.items()}
    
    # 다른 건 그대로 두고, 6번과 7번만 3000개로 맞춥니다.
    strategy[6] = max(counts[6], 3000)
    strategy[7] = max(counts[7], 3000)
    
    # 원본 데이터가 너무 적으므로 k_neighbors=3으로 낮춰서 안전하게 생성
    smote = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"✅ 스마트 증식 완료! (기존 {len(X_train)}개 -> 증식 후 {len(X_train_smote)}개)")
    print("가볍고 빠르게 학습이 진행됩니다 ⚡")

    # ==========================================
    # 🔬 2. Micro-Leaf LightGBM 모델
    # ==========================================
    print("\n🤖 Micro-Leaf LightGBM 모델 학습 시작... (미세 패턴 감지 모드)")
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=256,
        min_child_samples=5,  # 🔥 핵심: 데이터 5개만 모여도 잎사귀(결론) 생성 허용!
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_smote, y_train_smote)

    print("\n검증 데이터 예측 및 성능 평가 중... 📊")
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    print("\n========================================")
    print(f"🎯 핀셋 증식 + Micro-Leaf 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

    with open('../data/processed/precision_lgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장 완료! (data/processed/precision_lgbm_model.pkl)")

if __name__ == "__main__":
    train_precision_model()