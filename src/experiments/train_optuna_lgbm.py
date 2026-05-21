import numpy as np
import pickle
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore') # 보기 싫은 경고문 숨기기

def train_optuna_model():
    print("데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    # 3차원 데이터를 2차원으로 변환
    if len(X.shape) == 3:
        print(f"🔄 3차원 데이터를 2차원으로 변환: {X.shape} -> ", end="")
        X = X.reshape(X.shape[0], -1)
        print(f"변경 완료 {X.shape}")

    # 학습용 / 검증용 분리 (비율 유지)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Optuna가 수행할 단일 실험(Trial) 정의
    def objective(trial):
        # AI가 스스로 탐색해볼 튜닝 범위 설정 (학습률, 트리 깊이, 잎사귀 수 등)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', 7, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            # 소수 클래스 가중치를 줄지 말지도 AI가 직접 테스트해보고 결정하게 함
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        # 🔥 핵심: 단순 정확도(Accuracy)가 아니라, 모든 고장 모드를 골고루 잘 맞추는 
        # Macro F1 Score를 최적화 목표로 삼습니다. (6, 7번을 놓치면 점수가 깎임)
        score = f1_score(y_val, y_pred, average='macro')
        return score

    print("\n🧠 Optuna 하이퍼파라미터 자동 튜닝 시작!")
    print("AI가 총 20개의 각기 다른 세팅으로 모델을 학습시키며 최고점을 찾습니다.")
    
    # 점수가 높을수록 좋다고(maximize) 방향 설정
    study = optuna.create_study(direction='maximize')
    
    # 20번의 실험 진행 (나중에 더 완벽을 기하려면 n_trials=50, 100으로 늘리고 퇴근하시면 됩니다!)
    study.optimize(objective, n_trials=20)

    print("\n🏆 최적의 튜닝 파라미터 탐색 완료!")
    print(f"최고 Macro F1 점수: {study.best_value:.4f}")
    print("최적의 설정값:", study.best_params)

    print("\n🚀 찾아낸 최적의 설정값으로 최종 진화 모델 학습 중...")
    # 최고점을 기록한 세팅값으로 마지막 찐 학습
    best_model = lgb.LGBMClassifier(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)
    best_model.fit(X_train, y_train)

    print("최종 검증 데이터 예측 및 성능 평가 중... 📊")
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print("\n========================================")
    print(f"🎯 Optuna + LightGBM 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

    with open('../data/processed/optuna_lgbm_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("✅ 최적화된 모델 저장 완료! (data/processed/optuna_lgbm_model.pkl)")

if __name__ == "__main__":
    train_optuna_model()