# 🤖 서비스 로봇 실시간 예지보전(PdM) AI 파이프라인

## 📌 프로젝트 개요
실시간 서비스 로봇의 7개 다채널 센서 데이터를 분석하여 **10가지 상태(정상 작동 및 9가지 고장 모드)**를 0.01초 내에 예측하고, 비정상 진동 및 추세선을 감지하여 실시간 진단 결과를 반환하는 **예지보전(Predictive Maintenance, PdM) AI API 서버**입니다.

## 🚀 핵심 성과 및 엔지니어링 스토리

단순히 AI 알고리즘을 적용하는 것에 그치지 않고, **실제 제조/로봇 현장의 데이터적 한계를 공학적으로 돌파**하는 데 집중했습니다.

### 1. 초고속 추론을 위한 차원 압축 (3D ➡️ 2D Flatten)
* **Problem:** 30시점 연속(Sequence) 데이터를 딥러닝(RNN계열)으로 처리할 경우 실시간 관제 서버에 연산 과부하 발생 우려.
* **Solution:** 시계열 데이터를 2차원 평면으로 압축(Flatten)하여, 가볍고 빠른 트리 기반의 LightGBM 모델 도입.

### 2. SOTA 피처 엔지니어링 (시간 흐름 + 주파수 대역)
* **Problem:** 차원을 압축하면서 유실된 '시간의 흐름' 단서 때문에 주요 고장 예측률 정체.
* **Solution:** 장기 고장(배터리 저하 등)을 잡기 위한 **추세선(Drift) 피처**와 미세 진동을 잡아내기 위한 **고속 푸리에 변환(FFT) 주파수 피처**를 직접 수학적으로 추출하여 주입.

### 3. 극단적 불균형(Imbalance) 극복과 임계값(Threshold) 최적화
* **Problem:** 6번 고장(107건) vs 정상(20만 건)의 극단적 불균형. SMOTE 강제 증식 시 전체 정확도 붕괴 딜레마 발생.
* **Solution:** Optuna 하이퍼파라미터 튜닝 후, 희귀 고장 억지 학습(Balanced 족쇄)을 해제하여 **다수 클래스 집중형(Majority Focus)**으로 전환. 이후 소극적인 AI의 **결정 임계값을 35%로 최적화**하여 오보 리스크를 줄이면서 재현율(Recall) 극대화 ➡️ **🏆 최종 정확도 95.91% 달성**

### 4. 데이터 본질적 한계 증명 (EDA)
* `analyze_6.py` 스크립트를 통해 특정 희귀 고장(Class 6)이 정상 데이터의 노이즈 범위와 완벽히 중첩됨을 수학적으로 증명. 알고리즘의 문제가 아닌 '센서 한계'임을 규명하여 차기 이기종 센서(소음 등) 도입의 논리적 근거 마련.

## 🛠 기술 스택
- **AI/ML:** LightGBM, Optuna, Scikit-learn, Imbalanced-learn
- **Data Analysis:** Pandas, Numpy, FFT (Fast Fourier Transform)
- **Backend:** FastAPI, Uvicorn
- **Environment:** Python 3.11

## 📂 프로젝트 구조
```text
SERVICEROBOT_AI/
├── data/
│   └── processed/           # (GitIgnore) 대용량 전처리 데이터 및 최종 피클(.pkl) 모델 (보안/용량 상 별도 백업)
├── src/
│   ├── experiments/         # 💡 성능 한계 돌파를 위한 10여 개의 실험 아카이브 (SMOTE, Cascade, IsolationForest 등)
│   ├── analyze_6.py         # 센서 오버래핑(데이터 한계) 수학적 증명 EDA 스크립트
│   ├── preprocess.py        # 3차원 센서 데이터 전처리 (Flatten) 스크립트
│   ├── train_majority_focus.py     # 최종 SOTA 피처(FFT/Trend) 적용 LightGBM 학습
│   ├── train_majority_threshold.py # 결정 임계값(Threshold) 최적화로 95.9% 달성
│   ├── app.py               # FastAPI 기반 실시간 예지보전 추론 서버
│   └── client_test.py       # 로봇 시뮬레이션 및 API 통신 테스트
├── requirements.txt         # 패키지 의존성 명세
└── README.md                # 프로젝트 명세서

## 💻 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt

```

### 2. AI 추론 서버 실행

```bash
cd src
uvicorn app:app --reload

```

### 3. 클라이언트 통신 테스트 (새 터미널 열기)

```bash
cd src
python client_test.py

```


