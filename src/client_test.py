import requests
import numpy as np

def send_faulty_data_to_server():
    print("🤖 로봇: 센서 데이터를 수집 중입니다...")
    
    # 데이터와 정답지를 같이 부릅니다.
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train_multi.npy")
    
    # '정상(9번)'이 아닌 고장 난 데이터들의 인덱스만 싹 뽑아냅니다.
    faulty_indices = np.where(y_all != 9)[0]
    
    # 그 중 아무 고장 데이터나 하나 픽!
    sample_data = X_all[faulty_indices[0]] 
    
    flat_data_list = sample_data.flatten().tolist()
    
    url = "http://127.0.0.1:8000/predict"
    payload = {"data": flat_data_list}
    
    print(f"📡 로봇: 서버({url})로 진단을 요청합니다...")
    
    response = requests.post(url, json=payload)
    
    print("\n" + "="*40)
    print("🏢 서버로부터 도착한 진단 결과")
    print("="*40)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 상태: {result['status']}")
        print(f"🔎 원인: {result['error_code']}")
        print(f"📊 신뢰도: {result['confidence']}%")
    else:
        print(f"❌ 에러 발생: {response.status_code}")
    print("="*40)

if __name__ == "__main__":
    send_faulty_data_to_server()