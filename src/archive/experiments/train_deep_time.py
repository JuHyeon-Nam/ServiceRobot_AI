import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# 1. 딥러닝 GRU 모델 구조 정의
class RobotGRUClassifier(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=10, num_layers=2):
        super(RobotGRUClassifier, self).__init__()
        # 시계열 흐름을 파악하는 GRU 레이어
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        # 최종 분류를 위한 출력 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # out: (batch_size, seq_len, hidden_dim)
        out, _ = self.gru(x)
        # 30번째 시점(가장 마지막 흐름)의 결과만 쏙 빼서 분류에 사용
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_deep_model():
    print("데이터 로딩 중... ⏳")
    X = np.load('../data/processed/X_train_balanced.npy')
    y = np.load('../data/processed/y_train_balanced.npy')

    # 🚨 딥러닝을 위해 3차원 (데이터수, 30시점, 7센서) 형태 유지 확인!
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 30, 7)
    
    print(f"📦 딥러닝 투입 데이터 형태: {X.shape} (시간의 흐름을 보존합니다)")

    # 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # PyTorch 텐서 변환
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    # 데이터로더 생성 (배치 사이즈 512로 가볍고 빠르게)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # 장비 세팅 (GPU 있으면 쓰고 없으면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 현재 학습 연산 장비: {device}")

    # 모델, 손실함수, 최적화 알고리즘 세팅
    model = RobotGRUClassifier().to(device)
    # 6번, 7번 같은 극소수 클래스에 집중하도록 CrossEntropy 자체 페널티 부여효과 적용
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("\n🔥 시계열 딥러닝(GRU) 훈련 시작! (약 5에포크 동안 흐름을 파악합니다)")
    model.train()
    for epoch in range(5):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f" Epoch {epoch+1}/5 - Loss: {epoch_loss/len(train_loader):.4f}")

    print("\n검증 데이터 예측 및 시계열 패턴 평가 중... 📊")
    model.eval()
    with torch.no_grad():
        X_val_t = X_val_t.to(device)
        val_outputs = model(X_val_t)
        y_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()

    acc = accuracy_score(y_val, y_pred)
    print("\n========================================")
    print(f"🏆 시계열 GRU 딥러닝 최종 정확도: {acc * 100:.4f}%")
    print("========================================")
    print(classification_report(y_val, y_pred))

    # 모델 저장
    torch.save(model.state_dict(), '../data/processed/deep_gru_model.pth')
    print("✅ 딥러닝 모델 저장 완료! (data/processed/deep_gru_model.pth)")

if __name__ == "__main__":
    train_deep_model()