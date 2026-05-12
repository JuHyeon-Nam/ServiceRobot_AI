# 🤖 서비스 로봇 예지보전 다중 분류 AI 파이프라인

## 📌 프로젝트 개요
실시간 로봇 센서 데이터를 분석하여 10가지 상태(정상 및 9가지 고장 모드)를 예측하고, 실시간으로 진단 결과를 반환하는 AI API 서버입니다.

## 🚀 주요 성과
- **데이터 밸런싱 및 피처 엔지니어링:** 105만 건의 센서 시계열 데이터 전처리
- **모델 최적화:** LightGBM을 활용하여 극단적 데이터 불균형(Class Imbalance) 극복
- **최종 성능:** 다중 분류 정확도 **95.53%** 달성
- **API 서빙:** FastAPI를 활용한 실시간 예지보전 추론 서버 구축

## 🛠 기술 스택
- **AI/ML:** LightGBM, PyTorch, Scikit-learn, Pandas, Numpy
- **Backend:** FastAPI, Uvicorn
- **Environment:** Python 3.11

## 📂 프로젝트 구조
```text
ServiceRobot_AI/
├── data/
│   └── processed/           # (GitIgnore) 학습 모델 및 전처리 데이터
├── src/
│   ├── preprocess.py        # 데이터 전처리 및 라벨 인코딩
│   ├── train_lgbm.py        # LightGBM 다중 분류 모델 학습
│   ├── app.py               # FastAPI 실시간 추론 서버
│   ├── lgbm_real_test.py    # 로컬 예측 테스트
│   └── client_test.py       # 로봇 시뮬레이션 및 API 통신 테스트
├── requirements.txt         # 필요 라이브러리 목록
└── README.md                # 프로젝트 설명서

```

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


