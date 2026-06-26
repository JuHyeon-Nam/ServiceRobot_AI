import numpy as np
import pickle
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # 🔥 새로운 무기 추가

def train_ensemble_model():
    print("데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    if len(X.shape) == 3:
        print(f"🔄 3차원 데이터를 2차원으로 변환: {X.shape} -> ", end="")
        X = X.reshape(X.shape[0], -1)
        print(f"변경 완료 {X.shape}")

    # 1. 학습용 / 검증용 분리 (검증용은 건드리지 않아야 진짜 실력을 알 수 있음)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. 🔥 극단적 불균형 해결: SMOTE 적용
    print(f"✨ SMOTE 데이터 증식 시작... (기존 학습 데이터 개수: {len(X_train)}개)")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"✅ SMOTE 증식 완료! (증식된 학습 데이터 개수: {len(X_train_smote)}개)")

    # 3. 모델 세팅 (class_weight='balanced' 삭제!)
    print("개별 AI 모델 3대 세팅 중... 🤖")
    lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    xgboost = xgb.XGBClassifier(n_estimators=300, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    print("🔥 앙상블 모델(Soft Voting) 학습 시작! (데이터가 늘어나서 시간이 더 걸립니다 ☕)")
    voting_clf = VotingClassifier(
        estimators=[('lgbm', lgbm), ('xgb', xgboost), ('rf', rf)],
        voting='soft',
        weights=[2.5, 1.5, 1], # 가중치 살짝 조정
        n_jobs=-1
    )

    # 늘어난 가짜 데이터(SMOTE)로 학습
    voting_clf.fit(X_train_smote, y_train_smote)

    print("검증 데이터 예측 및 성능 평가 중... 📊")
    y_pred = voting_clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print("\n========================================")
    print(f"🏆 SMOTE 앙상블 모델 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

    with open('../data/processed/ensemble_multi_model.pkl', 'wb') as f:
        pickle.dump(voting_clf, f)
    print("✅ 모델 저장 완료!")

if __name__ == "__main__":
    train_ensemble_model()