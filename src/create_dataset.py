import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

def create_sequences(df, window_size=30):
    X, y = [], []
    
    # 로봇 ID별로 그룹화하여 데이터가 섞이지 않게 합니다.
    for device_id in tqdm(df['deviceId'].unique(), desc="Creating Sequences"):
        temp_df = df[df['deviceId'] == device_id].copy()
        
        # 모델 학습에 쓸 컬럼들 (수치형 데이터들)
        features = temp_df[['deviceType_encoded', 'batteryLevel', 'speed', 'x', 'y', 'collision', 'obstacle']].values
        target = temp_df['errorState'].values
        
        # 슬라이딩 윈도우 방식으로 묶기
        for i in range(len(temp_df) - window_size):
            X.append(features[i : i + window_size])
            # 윈도우 바로 다음 시점의 에러 여부를 정답으로 설정
            y.append(target[i + window_size])
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 1. 전처리된 데이터 로드
    df = pd.read_parquet("../data/processed/robot_preprocessed_data.parquet")
    
    # 2. 시퀀스 데이터 생성 (과거 30개 시점 사용)
    # 64GB RAM이므로 100만 건도 충분히 처리 가능합니다.
    X, y = create_sequences(df, window_size=30)
    
    print(f"✅ 시퀀스 생성 완료: X={X.shape}, y={y.shape}")
    
    # 3. 넘파이 배열로 저장 (용량이 커질 수 있으니 주의)
    # 나중에 모델 학습 코드에서 바로 불러올 수 있게 저장합니다.
    np.save("../data/processed/X_train.npy", X)
    np.save("../data/processed/y_train.npy", y)
    print("🎉 데이터셋 저장 완료!")