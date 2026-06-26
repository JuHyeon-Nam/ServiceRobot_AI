import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

def main():
    # 1. 데이터 로드 (언더샘플링 안 한 105만 개 원본 데이터!)
    print("⏳ 원본 다중 분류 데이터 로드 중...")
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train_multi.npy")

    # 2. 데이터 납작하게 누르기 (Flatten)
    # 3D (1055847, 30, 7) -> 2D (1055847, 210)
    print("🗜️ 3차원 시계열 데이터를 2차원으로 압축 중...")
    samples, timesteps, features = X_all.shape
    X_flat = X_all.reshape(samples, timesteps * features)

    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_all, test_size=0.2, random_state=42)

    # 4. LightGBM 모델 세팅 (여기서 마법이 일어납니다)
    print("🚀 LightGBM 모델 세팅 완료. 학습을 시작합니다!")
    clf = lgb.LGBMClassifier(
        n_estimators=1000,          # 나무를 1000개 심습니다.
        learning_rate=0.05,         # 꼼꼼하게 학습
        class_weight='balanced',    # 🔥 핵심: 희귀 고장에 엄청난 집중력을 부여!
        random_state=42,
        n_jobs=-1                   # CPU 코어 풀가동
    )

    # 5. 초고속 학습 시작
    start_time = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        # 조기 종료: 50번 동안 점수가 안 오르면 즉시 멈춤
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    print(f"⏱️ 학습 소요 시간: {time.time() - start_time:.1f}초")

    # 6. 정밀 성적표 출력
    print("\n🎯 테스트 데이터 예측 중...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    le = joblib.load("../data/processed/error_label_encoder.pkl")

    print("\n" + "="*50)
    print(f"🎉 LightGBM 최종 정확도: {acc * 100:.2f}%")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 모델 저장
    joblib.dump(clf, "../data/processed/lgbm_multi_model.pkl")
    print("💾 모델 저장 완료 (lgbm_multi_model.pkl)")

if __name__ == "__main__":
    main()