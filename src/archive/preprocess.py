import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

def preprocess_step():
    print("⏳ 데이터 불러오는 중...")
    # 1. 데이터 로드 (경로: src에서 실행 기준)
    df = pd.read_parquet("../data/processed/robot_total_data.parquet")
    
    # 2. 결측치 제거
    df = df.dropna()

    # 3. 로봇 종류(deviceType)를 숫자로 변환 (Label Encoding)
    le_device = LabelEncoder()
    df['deviceType_encoded'] = le_device.fit_transform(df['deviceType'])
    print(f"🤖 로봇 종류 매핑 완료: {list(le_device.classes_)}")

    # 4. 수치형 데이터 정규화 (Scaling)
    scaler = MinMaxScaler()
    cols_to_scale = ['batteryLevel', 'speed', 'x', 'y', 'collision', 'obstacle']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # ==========================================
    # 5. [추가] 고장 종류(errorCode) 다중 분류용 번역
    # ==========================================
    print("⚙️ 고장 종류(errorCode)를 다중 분류용 숫자(0~9)로 변환합니다...")
    le_error = LabelEncoder()
    
    # 'target_multi'라는 새로운 정답지 컬럼을 만듭니다.
    df['target_multi'] = le_error.fit_transform(df['errorCode'])
    
    # 나중에 숫자를 다시 고장 이름으로 바꾸기 위해 번역기를 저장해 둡니다.
    joblib.dump(le_error, "../data/processed/error_label_encoder.pkl")
    
    # 어떻게 번역되었는지 예쁘게 출력
    mapping_dict = dict(zip(le_error.classes_, le_error.transform(le_error.classes_)))
    print(f"✅ 고장 클래스 매핑 완료:\n{mapping_dict}")
    # ==========================================
    
    # 6. 결과 저장 (최종 전처리 데이터)
    df.to_parquet("../data/processed/robot_preprocessed_data.parquet", index=False)
    print("🎉 전처리 및 정규화 완료! (robot_preprocessed_data.parquet 저장됨)")

if __name__ == "__main__":
    preprocess_step()