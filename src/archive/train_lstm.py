import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

# ==========================================
# 1. 인공지능 '뇌' 구조 설계 (LSTM)
# ==========================================
class RobotLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super(RobotLSTM, self).__init__()
        # LSTM 층: 과거의 데이터를 기억하는 핵심 부품
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 출력층: 최종적으로 고장인지 아닌지 판단
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 형태: (배치크기, 30초, 7개정보)
        lstm_out, _ = self.lstm(x)
        # 마지막 30초 시점의 결과만 가져와서 판단 (가장 최신 흐름)
        last_time_step = lstm_out[:, -1, :] 
        out = self.fc(last_time_step)
        return self.sigmoid(out)

def main():
    print("⏳ 데이터 불러오는 중...")
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train.npy")
    
    # ⚠️ 중요: DNN과 달리 LSTM은 3차원(시간) 데이터를 그대로 씹어먹습니다!
    # X_all.reshape(...) 부분을 과감히 삭제했습니다.
    
    # 데이터 분리 (Train 70, Val 20, Test 10)
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # 파이토치 데이터셋 세팅
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    # 배치 사이즈 1024
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치 확인: {device}")

    # 모델, 오답노트, 최적화 도구 세팅
    model = RobotLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                val_loss += criterion(outputs, batch_y).item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                
        val_acc = correct / len(val_dataset) * 100
        print(f"Epoch [{epoch+1}/{epochs}] - {time.time() - start_time:.1f}초 | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), "../data/processed/lstm_model.pth")
    print("🎉 LSTM 모델 학습 및 저장 완료!")

if __name__ == "__main__":
    main()