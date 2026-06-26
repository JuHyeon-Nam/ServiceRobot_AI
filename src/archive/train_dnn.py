import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

def main():
    # ==========================================
    # 1. 데이터 로드 및 분리 (The Split)
    # ==========================================
    print("⏳ 데이터를 불러오는 중입니다... (최대 1~2분 소요)")
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train.npy")
    
    # DNN(기본 인공지능)은 2차원 데이터를 좋아합니다.
    # 현재 3차원(105만, 30, 7)인 X를 2차원(105만, 210)으로 쫙 펴줍니다.
    X_all = X_all.reshape(X_all.shape[0], -1) 
    
    # 1차 가위질: 학습용(70%) vs 나머지(30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    # 2차 가위질: 검증용(20%) vs 테스트용(10%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    print(f"📚 교과서(Train): {len(X_train)}개")
    print(f"📝 모의고사(Val): {len(X_val)}개")
    
    # ==========================================
    # 2. 파이토치(PyTorch) 데이터셋 세팅
    # ==========================================
    # GPU가 씹어먹기 좋게 Tensor(텐서) 형태로 변환합니다.
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    # 배치(Batch) 사이즈: 한 번에 AI에게 보여줄 문제의 개수
    # RTX 5060 Ti(16GB)는 성능이 좋아서 한 번에 1024개씩 보여줘도 거뜬합니다!
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # ==========================================
    # 3. 인공지능 '뇌' 구조 설계 (DNN)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치 확인: {device} (cuda가 떠야 GPU 풀가동 중인 겁니다!)")

    model = nn.Sequential(
        nn.Linear(210, 128), # 입력층 (30시점 * 7개 정보 = 210)
        nn.ReLU(),
        nn.Linear(128, 64),  # 은닉층
        nn.ReLU(),
        nn.Linear(64, 1),    # 출력층 (고장인지 아닌지 1개의 숫자로 출력)
        nn.Sigmoid()         # 결과를 0~100% 확률로 변환
    ).to(device)

    # 오답 노트(Loss)와 최적화 도구(Optimizer) 설정
    criterion = nn.BCELoss() # 이진 분류(고장/정상)에 쓰는 공식
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 4. 본격적인 학습 시작 (Training Loop)
    # ==========================================
    epochs = 10 # 교과서를 10번 반복해서 봅니다.
    
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
            
        # 모의고사(Validation) 평가
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                val_loss += criterion(outputs, batch_y).item()
                
                # 예측 확률이 0.5(50%)를 넘으면 고장(1)으로 판별
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                
        val_acc = correct / len(val_dataset) * 100
        time_taken = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] - {time_taken:.1f}초 소요 | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Accuracy(정확도): {val_acc:.2f}%")

    # 학습된 모델 저장
    torch.save(model.state_dict(), "../data/processed/dnn_baseline_model.pth")
    print("🎉 학습 완료! 모델이 저장되었습니다.")

if __name__ == "__main__":
    main()