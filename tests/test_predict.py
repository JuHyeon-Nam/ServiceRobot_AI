"""
PdM 추론 서버 핵심 불변식 테스트
- 피처 차원 정합(make_features 출력 = 모델 입력 249)  ← 과거 실제 디버깅 사례 방지
- /predict 응답 형태 + 설명가능성(reason) 검증
실행: (src 기준 경로를 쓰므로) `cd src && python -m pytest ../tests -q`  또는 repo 루트에서 `pytest -q`
"""
import os
import sys
import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
SRC = os.path.abspath(SRC)
sys.path.insert(0, SRC)


@pytest.fixture(scope="module")
def client():
    os.chdir(SRC)  # app.py가 상대경로(../data/processed)로 모델 로드
    from fastapi.testclient import TestClient
    import app
    return TestClient(app.app), app


def _window():
    rng = np.random.RandomState(0)
    return [[80 - t, 0.5, 1000 + t, 2000 + t, 90.0, 0, 0] for t in range(30)]


def test_group_index_matches_n_features():
    import explain
    g = explain.build_group_index(249)
    assert len(g) == 249, "피처 그룹 인덱스가 모델 피처 수(249)와 정합해야 함"


def test_feature_dim(client):
    _, app = client
    feats = app.make_features(_window(), app.Context())
    assert feats.shape == (1, 249), f"make_features는 (1,249)여야 함, got {feats.shape}"


def test_predict_shape_and_reason(client):
    cli, _ = client
    r = cli.post("/predict", json={"window": _window(), "context": {}})
    assert r.status_code == 200
    body = r.json()
    for k in ("error_code", "category", "confidence", "action_required", "reason"):
        assert k in body, f"응답에 {k} 누락"
    assert isinstance(body["reason"], list) and len(body["reason"]) == 3
    for item in body["reason"]:
        assert {"factor", "impact", "effect"} <= item.keys()


def test_model_card_endpoint(client):
    cli, _ = client
    r = cli.get("/model-card")
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == "robot-pdm-lightgbm-enhanced"
    assert body["feature_engineering"]["contract_ok"] is True
    assert len(body["artifact"]["sha256"]) == 64
    assert "/predict" in body["operations"]["serving_endpoints"]


def test_bad_input(client):
    cli, _ = client
    r = cli.post("/predict", json={"window": [[0] * 7] * 10})
    assert "error" in r.json()
