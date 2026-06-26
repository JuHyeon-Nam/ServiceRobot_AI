# 🤖 서비스 로봇 실시간 예지보전(PdM) AI 파이프라인

5종 서비스 로봇(안내·배송·서빙·물류·청소)의 다채널 센서 스트림을 분석해 **9가지 상태(정상 + 8종 고장)**를 실시간으로 예측하는 예지보전(Predictive Maintenance) AI 파이프라인입니다.

> **핵심 한 줄:** 무거운 RNN 대신 가벼운 LightGBM으로 0.01초 내 추론, 모델은 **2.8MB**로 어디서든 CPU만으로 동작. 그리고 *정확도 자랑이 아니라, 데이터의 한계를 공학적으로 규명하는 과정*을 담았습니다.

---

## 📊 최종 성능 (정직한 측정)

| 측정 방식 | 정확도 | macro-F1 | 비고 |
|---|---|---|---|
| **공식 Validation split** | **93.4%** | 0.62~0.72 | 데이터셋이 제공한 *학습에 안 쓰인* 검증셋. 처음 보는 로봇·시점으로 평가 → 신뢰 가능 |
| 무조건 '정상' 예측(기준선) | 83.0% | — | 모델이 이겨야 할 최소 기준 |

> ⚠️ **방법론 노트:** 초기 버전은 전체 데이터를 랜덤 분할해 95.9%가 나왔으나, *같은 로봇이 학습·평가에 동시에 들어가는* 약한 정보 누수가 있었습니다. 데이터셋의 **공식 Training/Validation split**(서로 다른 로봇)으로 재측정하여 **93.4%**라는 정직한 수치를 채택했습니다. 낮아 보이지만, 이것이 실제 배포 성능에 가깝습니다.

---

## 🚀 엔지니어링 스토리 (의사결정의 기록)

### 1. 왜 RNN을 버리고 LightGBM인가 — 실시간성
30시점 시퀀스를 GRU/LSTM(`src/archive/`에 실험 보관)으로 처리하면 정확하지만 관제 서버에 연산 부담. 시계열을 2D로 펴고(Flatten) 트리 기반 LightGBM으로 전환 → **추론 1ms, GPU 불필요, 모델 2.8MB**.

### 2. 왜 '버려진 피처'를 되살렸나 — 정답률의 진짜 레버
초기 파이프라인은 7개 필드만 사용했지만, 원본 JSON에는 마모·노화를 직접 가리키는 신호가 더 있었습니다. 이를 복원했더니 **피처 중요도 1·2·3위가 모두 복원한 피처**였습니다:

| 순위 | 피처 | 의미 |
|---|---|---|
| 1 | `distance` | 누적 주행거리 → 구동부 마모 |
| 2 | `batteryUse` | 누적 배터리 소모 → 배터리 노화 |
| 3 | `mainState` | 로봇 동작 상태(이동/충전 등) |

여기에 추세선(Drift)·FFT(주파수) 피처를 수학적으로 추출해 시간 흐름 단서를 보강했습니다.

### 3. 극단적 불균형 극복 — 언더샘플링 비율 탐색
정상 83% vs 일부 고장 수십 건의 극단적 불균형. SMOTE 강제 증식은 정확도를 붕괴시켜 폐기하고, 정상 클래스를 **에러 총합의 3배**로 언더샘플링하는 지점이 정확도·macro-F1을 동시에 최적화함을 실험으로 확정.

### 4. 데이터 본질적 한계의 규명 (EDA)
일부 고장(`E-RBT-N`, `E-RBT-S`)은 검증셋 표본이 1·14건에 불과하고 정상 데이터의 노이즈 범위와 중첩되어, **알고리즘이 아니라 센서의 한계**임을 규명(`analyze_6.py`). → 차기 이기종 센서(소음·전류 등) 도입의 논리적 근거.

---

## 🛠 기술 스택
- **AI/ML:** LightGBM, Scikit-learn, NumPy, FFT
- **Backend:** FastAPI, Uvicorn
- **Env:** Python 3.11 (CPU 추론, GPU 불필요)

## 📂 프로젝트 구조
```text
ServiceRobot_AI/
├── data/processed/                  # (대용량은 GitIgnore) 모델·전처리 산출물
│   ├── robot_pdm_enhanced.txt       # ✅ 최종 모델 (2.8MB, native LightGBM)
│   └── robot_pdm_enhanced_meta.json # 클래스·피처·인코딩 메타
├── src/
│   ├── build_enhanced_dataset.py    # 원본 zip → 강화 피처 + 공식 split 추출
│   ├── train_enhanced.py            # 최종 학습 + 공식 Validation 평가
│   ├── evaluate_portable.py         # 저장 모델 독립 로드 → 혼동행렬 측정
│   ├── app.py                       # FastAPI 실시간 추론 서버
│   ├── analyze_6.py                 # 센서 한계 규명 EDA
│   └── archive/                     # 초기 실험 아카이브(GRU/LSTM/DNN/SMOTE 등)
├── requirements.txt
└── README.md
```

## 💻 실행 방법
```bash
pip install -r requirements.txt

# 1) (선택) 원본 데이터로 강화 데이터셋 재생성  ※ 원본 zip 필요
cd src && python build_enhanced_dataset.py

# 2) (선택) 재학습 + 공식 Validation 평가
python train_enhanced.py

# 3) 저장된 모델로 실제 측정 테스트 (혼동행렬)
python evaluate_portable.py

# 4) 추론 API 서버 실행
uvicorn app:app --reload
```

> 📌 모델 파일(`robot_pdm_enhanced.txt`)만 있으면 학습 없이 어디서든 추론됩니다. 원본 1GB 데이터·전처리 `.npz`는 용량상 git에 올리지 않고 별도 보관합니다(아래).
