import numpy as np

def create_balanced_dataset():
    print("⏳ 원본 데이터 로드 중...")
    X = np.load("../data/processed/X_train.npy")
    y = np.load("../data/processed/y_train_multi.npy")

    # 정상 클래스의 인덱스 확인 (이전 로그 기준 '정상'은 9번)
    normal_class = 9

    normal_indices = np.where(y == normal_class)[0]
    error_indices = np.where(y != normal_class)[0]

    print(f"📊 원본 정상 데이터: {len(normal_indices)}개")
    print(f"📊 원본 고장 데이터 총합: {len(error_indices)}개")

    # 언더샘플링: 정상 데이터를 고장 데이터 수량과 동일하게 무작위 추출
    np.random.seed(42)
    sampled_normal_indices = np.random.choice(normal_indices, size=len(error_indices), replace=False)

    # 추출된 정상 데이터와 전체 고장 데이터를 결합
    balanced_indices = np.concatenate([sampled_normal_indices, error_indices])
    np.random.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    print(f"✅ 균형 조정 완료: X={X_balanced.shape}, y={y_balanced.shape}")

    # 새로운 이름으로 저장
    np.save("../data/processed/X_train_balanced.npy", X_balanced)
    np.save("../data/processed/y_train_balanced.npy", y_balanced)
    print("🎉 밸런싱 데이터셋 저장 완료!")

if __name__ == "__main__":
    create_balanced_dataset()