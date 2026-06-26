import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. 두 전문가의 뇌 구조 불러오기
# ==========================================
class RobotLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super(RobotLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

class RobotGRU(nn.Module):
    # 튜닝으로 찾았던 최적의 레시피 적용!
    def __init__(self, input_size=7, hidden_size=128, num_layers=1):
        super(RobotGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return self.sigmoid(out)

def main():
    print("⏳ 데이터 및 전문가 모델 불러오는 중...")
    # ==========================================
    # 2. 데이터 세팅 (한 번도 안 쓴 '진짜 수능 문제' Test 셋 사용)
    # ==========================================
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train.npy")
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 테스트 장치: {device}")

    # ==========================================
    # 3. 저장된 가중치(.pth) 주입
    # ==========================================
    model_lstm = RobotLSTM().to(device)
    model_lstm.load_state_dict(torch.load("../data/processed/lstm_model.pth", weights_only=True))
    model_lstm.eval()

    model_gru = RobotGRU().to(device)
    model_gru.load_state_dict(torch.load("../data/processed/best_gru_model.pth", weights_only=True))
    model_gru.eval()

    # ==========================================
    # 4. 앙상블 (Soft Voting) 채점 시작
    # ==========================================
    print("🤝 앙상블 투표 진행 중...")
    correct_ensemble = 0
    correct_lstm = 0
    correct_gru = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 각 모델의 예측 확률 계산
            prob_lstm = model_lstm(batch_x).squeeze()
            prob_gru = model_gru(batch_x).squeeze()
            
            # 앙상블: GRU(최적화)에 60%, LSTM에 40% 가중치 투표!
            prob_ensemble = (prob_gru * 0.6) + (prob_lstm * 0.4)
            
            # 0.5가 넘으면 고장(1)으로 판별
            pred_lstm = (prob_lstm > 0.5).float()
            pred_gru = (prob_gru > 0.5).float()
            pred_ensemble = (prob_ensemble > 0.5).float()
            
            correct_lstm += (pred_lstm == batch_y).sum().item()
            correct_gru += (pred_gru == batch_y).sum().item()
            correct_ensemble += (pred_ensemble == batch_y).sum().item()

    total = len(test_dataset)
    print("\n📊 === 최종 성적표 (완전 미지의 데이터) ===")
    print(f"🔹 LSTM 단독 정확도: {correct_lstm / total * 100:.2f}%")
    print(f"🔹 GRU 단독 정확도: {correct_gru / total * 100:.2f}%")
    print(f"🚀 앙상블 최종 정확도: {correct_ensemble / total * 100:.2f}%")

if __name__ == "__main__":
    main()