import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import optuna
import time

# ==========================================
# 1. 모델 설계도 (아까 빠졌던 부분!)
# ==========================================
class RobotGRU(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super(RobotGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_time_step = gru_out[:, -1, :] 
        out = self.fc(last_time_step)
        return self.sigmoid(out)

# ==========================================
# 2. 데이터 준비 (튜닝 속도를 위해 밖에서 한 번만 로드)
# ==========================================
print("⏳ 데이터 불러오는 중...")
X_all = np.load("../data/processed/X_train.npy")
y_all = np.load("../data/processed/y_train.npy")

X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

# RTX 5060 Ti의 VRAM(16GB)을 믿고 배치 사이즈를 늘려서 속도 극대화!
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. Optuna 튜닝 로직
# ==========================================
def objective(trial):
    # AI가 스스로 조합해 볼 경우의 수
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 2)

    model = RobotGRU(input_size=7, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 빠른 검증을 위해 조합당 딱 3번(Epoch)만 모의고사 실시
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 모의고사 채점
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()

    val_acc = correct / len(val_dataset) * 100
    return val_acc

# ==========================================
# 4. 본격 튜닝 시작
# ==========================================
if __name__ == "__main__":
    print(f"🚀 학습 장치 확인: {device}")
    
    # 방향: 정확도를 극대화(maximize)해라!
    study = optuna.create_study(direction="maximize")
    
    # 시간 절약을 위해 우선 10개의 조합만 테스트해 봅니다.
    print("🤖 10가지 하이퍼파라미터 조합 테스트를 시작합니다...")
    study.optimize(objective, n_trials=10) 

    print("\n🎉 최적화 완료!")
    print("🏆 찾은 최적의 파라미터:", study.best_params)
    print("🎯 이 조합으로 예상되는 최고 정확도:", study.best_value)