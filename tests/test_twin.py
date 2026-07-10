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


def test_agv_sensor_contract(client):
    """A4: 설비 탭 패널의 실시간 센서 그래프·AI 근거가 의존하는 계약."""
    s = client.get("/api/snapshot").json()
    a = s["agvs"][0]
    assert {"vib", "batt", "temp"} <= a["sensors"].keys(), "센서 텔레메트리 키 누락"
    for k in ("vib", "batt", "temp"):
        assert isinstance(a["sensors"][k], (int, float))
    assert 0 <= a["sensors"]["batt"] <= 100
    assert isinstance(a["cause"], list) and a["cause"], "AI 판단 근거(cause) 비면 안 됨"


def test_sensors_couple_to_fault(client):
    """진단(pred)에 센서가 물리적으로 커플링되는지: 배터리 저하→저배터리, 센서 이상→고진동."""
    from realtime_server import agv_sensors
    assert agv_sensors(10, 200, "E-RBT-B")["batt"] < 35, "배터리 저하인데 배터리 정상 수준"
    assert agv_sensors(10, 200, "E-RBT-S")["vib"] > 5, "센서 이상인데 진동 낮음"
    assert agv_sensors(10, 200, "정상")["batt"] > 35, "정상인데 배터리 저수준"


def test_alert_level_thresholds():
    """B2: 신뢰도(conf) → 경고 등급(주의/경고/위험) 트리아지 경계값."""
    from realtime_server import alert_level
    assert alert_level(0.95) == "위험"
    assert alert_level(0.85) == "위험"          # 경계 포함
    assert alert_level(0.70) == "경고"
    assert alert_level(0.60) == "경고"          # 경계 포함
    assert alert_level(0.40) == "주의"


def test_snapshot_level_contract(client):
    """B2: 관제 UI가 의존하는 경고 등급 계약 — AGV·alert·KPI 집계."""
    from realtime_server import LEVELS
    s = client.get("/api/snapshot").json()
    # 경고 상태 AGV는 등급을 갖고, 정상 AGV는 등급이 없다(None).
    for a in s["agvs"]:
        if a["status"] == "warn":
            assert a["level"] in LEVELS, f"경고 AGV에 등급 누락: {a['id']}"
        else:
            assert a["level"] is None
    for al in s["alerts"]:
        assert al["level"] in LEVELS
    by = s["kpi"]["by_level"]
    assert set(by.keys()) == set(LEVELS)
    assert sum(by.values()) == s["kpi"]["warn"], "등급별 합계가 총 경고 수와 불일치"


def test_offline_vendored_three(client):
    """A6: 전시장 인터넷 없이도 구동 — Three.js가 로컬 벤더링되어 /static에서 서빙되는지.
    import map이 CDN이 아니라 로컬 경로를 가리키고, 실제 파일이 200으로 서빙되어야 한다."""
    twin = client.get("/twin").text
    assert "unpkg.com" not in twin and "cdn" not in twin.lower(), "외부 CDN 참조가 남아 있음"
    assert "/static/vendor/three/build/three.module.js" in twin, "로컬 three import map 누락"
    # 실제 벤더 파일이 서빙되고 자바스크립트 모듈로 유효해야 함(에러 페이지가 아님).
    for path in ("/static/vendor/three/build/three.module.js",
                 "/static/vendor/three/examples/jsm/controls/OrbitControls.js"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} 서빙 실패"
        assert "from 'three'" in r.text or "REVISION" in r.text


def test_trend_history_contract(client):
    """B3: 설비 탭 타임라인이 의존하는 진단 추세(최근 N틱, 0~3 코드) 계약."""
    from realtime_server import TREND_W, diag_code
    s = client.get("/api/snapshot").json()
    a = s["agvs"][0]
    assert isinstance(a["trend"], list) and a["trend"], "진단 추세 이력이 비면 안 됨"
    assert len(a["trend"]) <= TREND_W
    assert all(isinstance(c, int) and 0 <= c <= 3 for c in a["trend"]), "추세 코드는 0~3"
    # 현재 상태와 마지막 추세 코드 정합: 정상이면 0, 이상이면 >0
    assert (a["trend"][-1] == 0) == (a["status"] == "ok")
    # diag_code 매핑: 정상=0, 위험(고신뢰)=3
    assert diag_code("정상", 0.99) == 0
    assert diag_code("E-RBT-B", 0.99) == 3


def test_trend_direction_drift(client):
    """B2: 악화 추세(드리프트) 조기 감지 — 방향 판정 + snapshot/KPI 계약."""
    from realtime_server import trend_direction
    assert trend_direction([0, 0, 0, 0, 2, 2, 3, 3]) == "악화"   # 뒤로 갈수록 나빠짐
    assert trend_direction([3, 3, 2, 2, 0, 0, 0, 0]) == "개선"   # 뒤로 갈수록 좋아짐
    assert trend_direction([1, 1, 1, 1, 1, 1]) == "안정"
    assert trend_direction([0, 0]) == "안정"                     # 너무 짧으면 안정
    s = client.get("/api/snapshot").json()
    assert all(a["trend_dir"] in ("악화", "개선", "안정") for a in s["agvs"])
    det = s["kpi"]["deteriorating"]
    assert isinstance(det, int)
    assert det == sum(a["trend_dir"] == "악화" for a in s["agvs"]), "악화 집계가 AGV별 방향과 불일치"


def test_health_index_and_maintenance(client):
    """B4: 자산 건전도 지표(Health Index) + 정비 우선순위 계약 — 관제 KPI·설비 패널이 의존."""
    from realtime_server import health_index, maint_advice
    # 정상·안정은 만점, 이상이 반복·고신뢰일수록 낮음(단조성)
    assert health_index([0, 0, 0, 0], 0.0, False) == 100
    assert health_index([3, 3, 3, 3], 0.99, True) < 40
    assert health_index([0, 0, 1, 2], 0.7, True) < health_index([0, 0, 0, 0], 0.0, False)
    assert all(2 <= health_index(t, 0.9, True) <= 100 for t in ([1], [2, 3], [0, 1, 2, 3]))
    # 권고는 건전도가 낮을수록 강한 조치
    assert maint_advice(95, "안정") == "정상 가동"
    assert "정비" in maint_advice(20, "악화")
    # snapshot 계약: 모든 AGV에 0~100 건전도 + 권고 문자열, KPI 집계 정합
    s = client.get("/api/snapshot").json()
    for a in s["agvs"]:
        assert 0 <= a["health"] <= 100 and isinstance(a["advice"], str) and a["advice"]
    assert s["kpi"]["maint_due"] == sum(a["health"] < 55 for a in s["agvs"]), "정비 집계 불일치"
    assert 0 <= s["kpi"]["avg_health"] <= 100


def test_telemetry_store_pipeline():
    """C4: 진단 이벤트 시계열 계층 — 선별 적재·이력 조회·롤업 집계·보존정책."""
    from telemetry_store import TelemetryStore
    st = TelemetryStore(":memory:", max_rows=5)
    agvs = [
        {"id": "AGV-01", "floor": 0, "status": "warn", "pred": "E-RBT-B", "conf": 0.9,
         "level": "위험", "health": 30, "sensors": {"vib": 7, "batt": 20, "temp": 55}},
        {"id": "AGV-02", "floor": 1, "status": "ok", "pred": "정상", "conf": 0.99,
         "level": None, "health": 100, "sensors": {"vib": 2, "batt": 80, "temp": 38}},
    ]
    assert st.record(100.0, agvs) == 1          # 정상·건전100은 스킵, 이상 1건만 적재
    st.record(101.0, agvs)
    hist = st.history("AGV-01", 10)
    assert len(hist) == 2 and hist[0]["pred"] == "E-RBT-B"   # 최신순
    stats = st.stats()
    assert stats["total"] == 2 and stats["by_level"].get("위험") == 2
    assert stats["top_agv"][0]["agv"] == "AGV-01"
    assert stats["by_floor"] == {"0": 2}
    for t in range(10):                          # 보존정책: max_rows 초과분 삭제
        st.record(200.0 + t, agvs)
    st.prune()
    assert st.stats()["total"] <= 5


def test_telemetry_endpoints(client):
    """C4: 시계열 데이터 계층 조회 API 계약(/api/stats · /api/history)."""
    s = client.get("/api/stats").json()
    assert {"total", "by_level", "by_floor", "top_agv", "avg_health"} <= s.keys()
    assert isinstance(s["top_agv"], list)
    r = client.get("/api/history", params={"agv": "AGV-01", "limit": 10})
    assert r.status_code == 200 and isinstance(r.json(), list)
