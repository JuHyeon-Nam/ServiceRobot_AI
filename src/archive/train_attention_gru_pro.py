import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

# [모델 설계도는 이전과 동일하게 AttentionGRU 유지]
class AttentionGRU(nn.Module):
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

# 🔥 [추가] 97%를 위한 최종 비기: Focal Loss 구현
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("⏳ 데이터 로드 및 치료(Clipping) 중...")
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train_multi.npy") 
    
    # ==========================================
    # 🔥 STEP 1: 벌점 상한선(Clipping) 적용!
    # ==========================================
    class_counts = np.bincount(y_all.astype(int))
    total_samples = len(y_all)
    weights = total_samples / (len(class_counts) * class_counts)
    
    # 986배 벌점을 최대 15배까지만 허용합니다! (숫자는 조절 가능)
    weights = np.clip(weights, a_min=None, a_max=15.0) 
    weights = torch.FloatTensor(weights).to(device)
    print(f"⚖️ 치료된 가중치 (최대 15배):\n{weights}")
    # ==========================================

    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    model = AttentionGRU(input_size=7, hidden_size=128, num_layers=1, num_classes=10).to(device)
    
    # 🔥 STEP 3: Focal Loss 적용! (기존 CrossEntropy 대신 사용)
    criterion = FocalLoss(weight=weights, gamma=2.0)
    
    # AI가 차분해지도록 학습 보폭(lr)을 줄입니다.
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    epochs = 30
    best_acc = 0

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                
        val_acc = correct / len(val_dataset) * 100
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "../data/processed/attention_gru_focal.pth")
            mark = "🌟 안정화 성공!"
        else:
            mark = ""

        print(f"Epoch [{epoch+1}/{epochs}] - {time.time()-start_time:.1f}초 | 정확도: {val_acc:.2f}% {mark}")

    print(f"\n🎉 치료 완료 후 최종 정확도: {best_acc:.2f}%")

if __name__ == "__main__":
    main()