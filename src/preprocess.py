import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_step():
    # 1. 데이터 로드 (경로: src에서 실행 기준)
    df = pd.read_parquet("../data/processed/robot_total_data.parquet")
    
    # 2. 결측치 제거
    df = df.dropna()

    # 3. 로봇 종류(deviceType)를 숫자로 변환 (Label Encoding)
    # 예: 안내로봇 -> 0, 배송로봇 -> 1 ...
    le = LabelEncoder()
    df['deviceType_encoded'] = le.fit_transform(df['deviceType'])
    print(f"🤖 로봇 종류 매핑 완료: {list(le.classes_)}")

    # 4. 수치형 데이터 정규화 (Scaling)
    scaler = MinMaxScaler()
    cols_to_scale = ['batteryLevel', 'speed', 'x', 'y', 'collision', 'obstacle']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # 5. 결과 저장 (최종 전처리 데이터)
    df.to_parquet("../data/processed/robot_preprocessed_data.parquet", index=False)
    print("🎉 전처리 및 정규화 완료!")

if __name__ == "__main__":
    preprocess_step()