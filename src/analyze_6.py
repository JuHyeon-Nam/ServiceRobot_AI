import numpy as np
import pandas as pd

def analyze_minority_class():
    print("데이터 분석 로딩 중... ⏳")
    # 원본 3차원 모양 그대로 확인하기 위해 원본을 상상하며 변환
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    # 만약 납작해진 상태라면 3차원(30, 7)으로 복원해서 센서별 특징 확인
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 30, 7)

    print("\n==================================================")
    print("🔍 [데이터 본질 분석] 9번(정상) vs 6번(희귀 고장) 비교")
    print("==================================================")

    # 9번 정상과 6번 고장 인덱스 추출
    idx_9 = (y == 9)
    idx_6 = (y == 6)

    X_9 = X[idx_9]
    X_6 = X[idx_6]

    print(f"• 전체 데이터 중 9번(정상) 개수: {len(X_9)}개")
    print(f"• 전체 데이터 중 6번(고장) 개수: {len(X_6)}개")
    
    if len(X_6) == 0:
        print("🚨 6번 데이터가 학습 셋에 존재하지 않습니다! 데이터 확인 필요.")
        return

    # 30개 시점(시간)에 대해 평균을 내어, 7개 센서의 평균값 비교
    mean_9 = np.mean(X_9, axis=(0, 1))
    mean_6 = np.mean(X_6, axis=(0, 1))
    
    std_9 = np.std(X_9, axis=(0, 1))
    std_6 = np.std(X_6, axis=(0, 1))

    # 표로 예쁘게 만들기
    analysis_df = pd.DataFrame({
        '정상(9번) 평균': mean_9,
        '고장(6번) 평균': mean_6,
        '평균 차이': np.abs(mean_9 - mean_6),
        '정상(9번) 편차': std_9,
        '고장(6번) 편차': std_6
    })
    
    analysis_df.index = [f'센서_{i+1}' for i in range(7)]
    
    print("\n[7개 센서의 물리적 수치 비교 표]")
    print(analysis_df.to_string())
    print("\n==================================================")
    
    # 분석 가이드 출력
    min_diff = np.min(np.abs(mean_9 - mean_6))
    max_diff = np.max(np.abs(mean_9 - mean_6))
    
    print(f"💡 분석 결과 힌트:")
    print(f" - 센서 간 최대 평균 차이: {max_diff:.6f}")
    print(f" - 센서 간 최소 평균 차이: {min_diff:.6f}")
    print("\n만약 '평균 차이'가 모든 센서에서 0에 가깝다면,")
    print("AI가 아니라 그 어떤 최신 알고리즘을 가져와도 구분이 불가능한 데이터 상태입니다.")
    print("==================================================")

if __name__ == "__main__":
    analyze_minority_class()