import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

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
    print(f"🚀 학습 장치: {device}")
    
    # 원본 다중 분류 데이터 (밸런싱 이전의 전체 데이터)를 부릅니다.
    print("⏳ 원본 데이터 로드 중...")
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train_multi.npy") 
    
    # ==========================================
    # 🔥 [핵심] '정상(9번)' 데이터를 완전히 삭제합니다.
    # 오직 고장 데이터만 가지고 학습합니다.
    # ==========================================
    error_indices = np.where(y_all != 9)[0]
    X_errors = X_all[error_indices]
    y_errors = y_all[error_indices]
    
    print(f"📊 고장 전용 데이터 필터링 완료: {len(y_errors)}개")
    
    # 가중치 계산 (극단적인 페널티 방지를 위해 최대 5배까지만 허용)
    class_counts = np.bincount(y_errors.astype(int), minlength=10)
    
    # 0인 클래스(정상)는 가중치 계산에서 오류를 내므로 1로 임시 대체
    class_counts[class_counts == 0] = 1 
    
    weights = len(y_errors) / (9 * class_counts) # 9개 고장 클래스 기준
    weights[9] = 0.0 # 정상(9번)은 학습하지 않으므로 가중치 0
    weights = np.clip(weights, a_min=None, a_max=5.0) 
    weights = torch.FloatTensor(weights).to(device)
    
    print(f"⚖️ 클리핑된 고장 전용 가중치:\n{weights}")

    X_train, X_temp, y_train, y_temp = train_test_split(X_errors, y_errors, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    model = AttentionGRU(input_size=7, hidden_size=128, num_layers=1, num_classes=10).to(device)
    
    # 클리핑된 가중치를 적용한 CrossEntropy
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    best_acc = 0

    print("🎯 [고장 종류 전용 분류기(Specialist)] 학습 시작...")
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
            torch.save(model.state_dict(), "../data/processed/error_specialist.pth")
            mark = "🌟 최고 성능 갱신!"
        else:
            mark = ""

        print(f"Epoch [{epoch+1}/{epochs}] - {time.time()-start_time:.1f}초 | 정확도: {val_acc:.2f}% {mark}")

if __name__ == "__main__":
    main()