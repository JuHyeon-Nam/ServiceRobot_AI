"""
3D 디지털 트윈이 의존하는 실시간 서버 계약(contract) 테스트.
- /twin 페이지 서빙
- /api/layout 구조(층·장비·트랙) — twin.html 빌드가 기대하는 키
- /api/snapshot 구조(AGV 키·KPI) — twin.html applyState가 기대하는 키
이 계약이 깨지면 3D 화면이 조용히 망가지므로 회귀를 잠근다.
실행: cd src && python -m pytest ../tests -q
"""
import os
import sys
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)


@pytest.fixture(scope="module")
def client():
    os.chdir(SRC)
    from fastapi.testclient import TestClient
    import realtime_server
    return TestClient(realtime_server.app)


def test_twin_page_served(client):
    r = client.get("/twin")
    assert r.status_code == 200
    assert "디지털 트윈" in r.text and "three" in r.text.lower()


def test_layout_contract(client):
    L = client.get("/api/layout").json()
    assert L["canvas"] == {"w": 300, "h": 196}
    assert len(L["floors"]) == 3
    f0 = L["floors"][0]
    for key in ("y0", "short", "name", "geo", "tracks", "tools"):
        assert key in f0
    assert f0["tools"] and {"cx", "cy", "w", "h"} <= f0["tools"][0].keys()


def test_snapshot_contract(client):
    s = client.get("/api/snapshot").json()
    assert s["type"] == "state"
    assert {"total", "ok", "warn", "per_floor"} <= s["kpi"].keys()
    assert s["agvs"], "AGV 목록이 비면 안 됨"
    a = s["agvs"][0]
    for key in ("id", "x", "y", "ang", "floor", "status", "pred", "label", "conf"):
        assert key in a, f"AGV 응답에 {key} 누락 (3D 렌더가 의존)"
    assert a["status"] in ("ok", "warn")
