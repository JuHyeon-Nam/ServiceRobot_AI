import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

class RobotGRU(nn.Module):
    # 🏆 Optuna가 찾은 최적의 세팅: hidden_size=128, num_layers=1
    def __init__(self, input_size=7, hidden_size=128, num_layers=1):
        super(RobotGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_time_step = gru_out[:, -1, :] 
        out = self.fc(last_time_step)
        return self.sigmoid(out)

def main():
    print("⏳ 데이터 불러오는 중...")
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train.npy")
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치 확인: {device} (풀파워 가동!)")

    model = RobotGRU().to(device)
    criterion = nn.BCELoss()
    
    # 🏆 Optuna가 찾은 최적의 학습률: 약 0.004
    optimizer = optim.Adam(model.parameters(), lr=0.004)

    # 이번엔 3번이 아니라 충분히 20번을 돌려봅니다!
    epochs = 20
    best_acc = 0

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
        
        # 신기록을 세울 때마다 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "../data/processed/best_gru_model.pth")
            mark = "🌟 신기록!"
        else:
            mark = ""

        print(f"Epoch [{epoch+1}/{epochs}] - {time.time() - start_time:.1f}초 | Loss: {val_loss/len(val_loader):.4f} | Accuracy: {val_acc:.2f}% {mark}")

    print(f"\n🎉 최종 최고 정확도: {best_acc:.2f}%")

if __name__ == "__main__":
    main()