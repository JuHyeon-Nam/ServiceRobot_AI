import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

def create_sequences(df, window_size=30):
    X, y = [], []
    
    # 로봇 ID별로 그룹화하여 데이터가 섞이지 않게 합니다.
    unique_devices = df['deviceId'].unique()
    for device_id in tqdm(unique_devices, desc="Creating Sequences"):
        temp_df = df[df['deviceId'] == device_id].copy()
        
        # 모델 학습에 쓸 컬럼들 (수치형 데이터들)
        features = temp_df[['deviceType_encoded', 'batteryLevel', 'speed', 'x', 'y', 'collision', 'obstacle']].values
        
        # ==========================================
        # [수정] 다중 분류용 정답(target_multi)을 가져옵니다.
        # ==========================================
        target = temp_df['target_multi'].values 
        
        # 슬라이딩 윈도우 방식으로 묶기
        if len(temp_df) <= window_size:
            continue

        for i in range(len(temp_df) - window_size):
            X.append(features[i : i + window_size])
            # 윈도우 바로 다음 시점의 '고장 코드(0~9)'를 정답으로 설정
            y.append(target[i + window_size])
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 1. 전처리된 데이터 로드 (preprocess.py에서 만든 파일)
    print("⏳ 전처리된 데이터 불러오는 중...")
    df = pd.read_parquet("../data/processed/robot_preprocessed_data.parquet")
    
    # 2. 시퀀스 데이터 생성 (과거 30개 시점 사용)
    window_size = 30
    X, y = create_sequences(df, window_size=window_size)
    
    print(f"✅ 시퀀스 생성 완료: X={X.shape}, y={y.shape}")
    
    # 3. 넘파이 배열로 저장
    # X_train은 그대로 두고, 정답지(y)만 다중 분류용 이름으로 구분해서 저장합니다.
    np.save("../data/processed/X_train.npy", X)
    np.save("../data/processed/y_train_multi.npy", y) # 파일명 변경!
    
    print("🎉 다중 분류용 데이터셋 저장 완료! (y_train_multi.npy)")