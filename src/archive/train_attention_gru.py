import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

# [모델 설계도는 동일하므로 생략하거나 이전 코드의 AttentionGRU를 그대로 사용하세요]
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("⏳ 다중 분류 데이터 로드 중...")
    # X_all = np.load("../data/processed/X_train.npy")
    # y_all = np.load("../data/processed/y_train_multi.npy") 
    X_all = np.load("../data/processed/X_train_balanced.npy")
    y_all = np.load("../data/processed/y_train_balanced.npy")


    # ==========================================
    # 🔥 [핵심] 가중치 계산 (Class Weights)
    # 데이터가 적은 고장일수록 더 큰 벌점을 부여합니다.
    # ==========================================
    class_counts = np.bincount(y_all.astype(int))
    total_samples = len(y_all)
    # 가중치 공식: 전체 샘플 / (클래스 수 * 해당 클래스 샘플 수)
    weights = total_samples / (len(class_counts) * class_counts)
    weights = torch.FloatTensor(weights).to(device)
    print(f"⚖️ 적용된 클래스별 가중치:\n{weights}")
    # ==========================================

    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    model = AttentionGRU(input_size=7, hidden_size=128, num_layers=1, num_classes=10).to(device)
    
    # 가중치 적용된 손실 함수!
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.002) # 학습률을 살짝 낮춰 정밀하게 학습

    epochs = 30 # 좀 더 끈기 있게 학습
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
            torch.save(model.state_dict(), "../data/processed/attention_gru_final.pth")
            mark = "🌟 97%를 향한 신기록!"
        else:
            mark = ""

        print(f"Epoch [{epoch+1}/{epochs}] - {time.time()-start_time:.1f}초 | 정확도: {val_acc:.2f}% {mark}")

    print(f"\n🎉 가중치 적용 최종 최고 정확도: {best_acc:.2f}%")

if __name__ == "__main__":
    main()