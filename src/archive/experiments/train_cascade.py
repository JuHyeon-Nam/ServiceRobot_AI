import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_cascade_model():
    print("데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    if len(X.shape) == 3:
        print(f"🔄 3차원 데이터를 2차원으로 변환: {X.shape} -> ", end="")
        X = X.reshape(X.shape[0], -1)
        print(f"변경 완료 {X.shape}")

    # 데이터 분리 (검증용 20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # 🎯 Stage 1: 정상(9) vs 고장(나머지) 이진 분류
    # ==========================================
    NORMAL_CLASS = 9  # 정상 클래스 번호 (필요시 수정)
    print(f"\n[Stage 1] 정상({NORMAL_CLASS}번) vs 고장 분류 모델 학습 중... 🤖")
    
    # 9번이면 0(정상), 아니면 1(고장)로 라벨 변경
    y_train_stage1 = (y_train != NORMAL_CLASS).astype(int) 
    y_val_stage1 = (y_val != NORMAL_CLASS).astype(int)

    # 1단계 모델: 극강의 정확도를 위해 트리를 깊게 씁니다.
    model_stage1 = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model_stage1.fit(X_train, y_train_stage1)

    # ==========================================
    # 🔎 Stage 2: 고장 유형(0~8) 다중 분류 (정상 데이터 배제)
    # ==========================================
    print("\n[Stage 2] 고장 유형 정밀 진단 모델 학습 중... 🔬 (6, 7번 구출 작전!)")
    
    # 학습 데이터에서 '고장'인 것들만 솎아내기
    error_indices = (y_train != NORMAL_CLASS)
    X_train_stage2 = X_train[error_indices]
    y_train_stage2 = y_train[error_indices]

    # 2단계 모델: 정상 데이터가 빠졌으니, 남은 희귀 고장에 집중하도록 'balanced' 부여
    model_stage2 = lgb.LGBMClassifier(n_estimators=400, class_weight='balanced', random_state=42, n_jobs=-1)
    model_stage2.fit(X_train_stage2, y_train_stage2)

    # ==========================================
    # 🚀 최종 예측 파이프라인 (검증)
    # ==========================================
    print("\n검증 데이터로 2단계 캐스케이드 추론 진행 중... 📊")
    
    # 1. 일단 모두 정상(9)이라고 가짜 정답지 생성
    final_predictions = np.full(shape=len(X_val), fill_value=NORMAL_CLASS)

    # 2. Stage 1 모델로 전체 데이터를 보고 고장(1) 여부 판별
    stage1_preds = model_stage1.predict(X_val)
    
    # 3. 고장이라고 판정된 데이터들의 인덱스만 뽑기
    predicted_errors_mask = (stage1_preds == 1)

    # 4. 고장으로 의심되는 놈들만 Stage 2 모델에 넣어서 정확한 병명 진단
    if np.sum(predicted_errors_mask) > 0:
        stage2_preds = model_stage2.predict(X_val[predicted_errors_mask])
        final_predictions[predicted_errors_mask] = stage2_preds

    # ==========================================
    # 🏆 결과 확인
    # ==========================================
    acc = accuracy_score(y_val, final_predictions)
    print("\n========================================")
    print(f"🎯 2단계 캐스케이드 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, final_predictions))

    # 파이프라인 모델 저장 (두 개의 모델을 딕셔너리로 묶어서 저장)
    cascade_pipeline = {'stage1': model_stage1, 'stage2': model_stage2, 'normal_class': NORMAL_CLASS}
    with open('../data/processed/cascade_lgbm_model.pkl', 'wb') as f:
        pickle.dump(cascade_pipeline, f)
    print("✅ 캐스케이드 모델 저장 완료! (data/processed/cascade_lgbm_model.pkl)")

if __name__ == "__main__":
    train_cascade_model()