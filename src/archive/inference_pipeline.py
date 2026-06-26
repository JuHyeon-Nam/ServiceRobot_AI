import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# 1. 두 전문가의 설계도 불러오기
class RobotGRU(nn.Module): # Stage 1 (고장 판독기)
    def __init__(self, input_size=7, hidden_size=128, num_layers=1):
        super(RobotGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.sigmoid(self.fc(gru_out[:, -1, :]))

class AttentionGRU(nn.Module): # Stage 2 (고장 원인 분석기)
    def __init__(self, input_size=7, hidden_size=128, num_layers=1, num_classes=10):
        super(AttentionGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        gru_out, _ = self.gru(x) 
        attn_weights = F.softmax(self.attention(gru_out), dim=1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)
        return self.fc(context_vector)

def predict_robot_status(sensor_data_30s):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 및 번역기 로드
    stage1_model = RobotGRU().to(device)
    stage1_model.load_state_dict(torch.load("../data/processed/best_gru_model.pth", weights_only=True))
    stage1_model.eval()

    stage2_model = AttentionGRU().to(device).to(device)
    stage2_model.load_state_dict(torch.load("../data/processed/error_specialist.pth", weights_only=True))
    stage2_model.eval()
    
    le = joblib.load("../data/processed/error_label_encoder.pkl")

    # 추론 시작
    input_tensor = torch.FloatTensor(sensor_data_30s).unsqueeze(0).to(device) # (1, 30, 7) 형태로 변환

    with torch.no_grad():
        # [Stage 1] 고장 유무 확인
        is_fault_prob = stage1_model(input_tensor).item()
        
        if is_fault_prob < 0.5:
            return "✅ 정상 작동 중입니다. (고장 확률: {:.1f}%)".format(is_fault_prob * 100)
        
        # [Stage 2] 고장일 경우, 원인 파악
        print(f"⚠️ 이상 징후 감지! (고장 확률: {is_fault_prob * 100:.1f}%) -> 정밀 분석을 시작합니다.")
        specialist_out = stage2_model(input_tensor)
        
        # 가장 확률이 높은 고장 코드 번호 추출
        _, predicted_idx = torch.max(specialist_out, 1)
        error_code_str = le.inverse_transform([predicted_idx.item()])[0]
        
        return f"🚨 고장 예측: 30초 뒤 [{error_code_str}] 에러가 발생할 것으로 예상됩니다."

if __name__ == "__main__":
    # 테스트용: X_train 데이터 중 아무거나 하나 뽑아서 로봇이 보낸 실시간 데이터라고 가정해 봅시다.
    print("⏳ 파이프라인 초기화 중...")
    X_test = np.load("../data/processed/X_train.npy")
    
    print("\n--- 🤖 로봇 1번 실시간 데이터 수신 ---")
    # 정상일 확률이 높은 데이터 세트 (테스트)
    sample_data_1 = X_test[0] 
    print(predict_robot_status(sample_data_1))

    print("\n--- 🤖 로봇 2번 실시간 데이터 수신 ---")
    # 고장일 확률이 높은 데이터 세트 (테스트) - 임의로 인덱스 800000번 대를 뽑아봅니다
    sample_data_2 = X_test[-1] 
    print(predict_robot_status(sample_data_2))