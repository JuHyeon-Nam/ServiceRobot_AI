# FAB 3D 디지털 트윈 · 실시간 PdM 관제 서버 — 한 줄 배포:
#   docker build -t servicerobot-ai . && docker run -p 8000:8000 servicerobot-ai
#   → http://127.0.0.1:8000/twin
# 이미지는 관제 서버 전용 최소 의존성(requirements-server.txt)만 설치해 경량 유지.
# CI(.github/workflows/ci.yml)가 매 push마다 빌드 + /twin·/api·/metrics 스모크 테스트로 검증.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY data/processed/ data/processed/
COPY src/ src/

ENV PYTHONUTF8=1
EXPOSE 8000

WORKDIR /app/src
CMD ["uvicorn", "realtime_server:app", "--host", "0.0.0.0", "--port", "8000"]
