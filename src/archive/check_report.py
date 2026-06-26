import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 모델 설계도 (파이썬은 이 설계도가 꼭 필요합니다)
# ==========================================
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

def check_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 분석 장치: {device}")
    
    # 1. 데이터 및 번역기(LabelEncoder) 로드
    print("⏳ 데이터 및 모델 로드 중...")
    X_all = np.load("../data/processed/X_train.npy")
    y_all = np.load("../data/processed/y_train_multi.npy")
    le = joblib.load("../data/processed/error_label_encoder.pkl")
    
    # 테스트 데이터만 분리 (학습 때와 동일한 seed 사용)
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42)
    
    # 2. 모델 불러오기
    model = AttentionGRU(input_size=7, hidden_size=128, num_layers=1, num_classes=10).to(device)
    model.load_state_dict(torch.load("../data/processed/attention_gru_multi.pth", weights_only=True))
    model.eval()

    # 3. 예측 진행 (10만 건 이상이므로 배치 단위로 실행)
    all_preds = []
    print("🎯 고장 종류별 정밀 분석 시작...")
    with torch.no_grad():
        for i in range(0, len(X_test), 2048):
            batch_x = torch.FloatTensor(X_test[i:i+2048]).to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    # 4. 상세 성적표 출력
    print("\n" + "="*50)
    print("📝 [서비스 로봇 고장 예측 AI 상세 성적표]")
    print("="*50)
    # 각 고장 코드별로 정밀도, 재현율, F1-score를 보여줍니다.
    print(classification_report(y_test, all_preds, target_names=le.classes_))
    print("="*50)

if __name__ == "__main__":
    check_performance()