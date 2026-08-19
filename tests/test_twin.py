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
import time
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


def test_twin_phm_patrol_cues(client):
    r = client.get("/twin")
    assert r.status_code == 200
    assert "순찰 모드" in r.text
    assert "PHM 단계" in r.text
    assert "PHM 위험도" in r.text
    assert "예상 대응시점" in r.text
    assert "운영 Dispatch" in r.text
    assert "selImpactBar" in r.text
    assert "selWorkOrder" in r.text
    assert "Edge 입력" in r.text
    assert "데이터 출처" in r.text
    assert "sourceStat" in r.text
    assert "focus = agvMesh" in r.text
    assert "#sel { position:fixed" in r.text
    assert "#title p { display:none; }" in r.text
    assert "phmStage" in r.text
    assert "autoRotate" in r.text


def test_demo_hub_page_served(client):
    r = client.get("/demo")
    assert r.status_code == 200
    assert "ServiceRobot_AI Demo Hub" in r.text
    for expected in ("/twin", "/api/phm", "/api/tsdb-export?fmt=influx",
                     "/api/edge-ingest", "/api/ops-report?fmt=md",
                     "/api/model-card", "/assets/twin_3d.gif"):
        assert expected in r.text


def test_demo_assets_served(client):
    r = client.get("/assets/twin_3d.gif")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/gif")


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
    assert s["inference"]["mode"] == "live_booster"
    assert s["inference"]["n_features"] == 249
    assert s["inference"]["calls"] > 0
    assert {"dataset", "runtime", "rule_based_parts", "model_based_parts", "edge_active"} <= s["data_source"].keys()
    assert s["agvs"], "AGV 목록이 비면 안 됨"
    a = s["agvs"][0]
    for key in ("id", "x", "y", "ang", "floor", "status", "pred", "label", "conf"):
        assert key in a, f"AGV 응답에 {key} 누락 (3D 렌더가 의존)"
    assert a["status"] in ("ok", "warn")
    assert {"stage", "severity", "risk_score", "rul_estimate_min", "reasons", "action"} <= a["phm"].keys()
    assert 0 <= a["phm"]["risk_score"] <= 100
    assert {"state", "priority", "sla_min", "impact_pct", "affected_zone", "route_block_risk",
            "work_order_required", "operator_action"} <= a["dispatch"].keys()
    assert 0 <= a["dispatch"]["impact_pct"] <= 100


def test_data_source_endpoint_exposes_demo_boundaries(client):
    """시연 replay, 실제 모델 추론, 규칙 기반 PHM의 경계를 API로 확인할 수 있어야 한다."""
    body = client.get("/api/data-source").json()
    assert body["dataset"].startswith("AI-Hub")
    assert body["physical_robot_connected"] is False
    assert body["replay_motion"]["estimated_cycle_sec"] >= 10
    assert "PHM risk/RUL heuristic" in body["rule_based_parts"]
    assert "9-class fault diagnosis" in body["model_based_parts"]

    import realtime_server
    realtime_server.EDGE_INPUTS["AGV-EDGE-TEST"] = {"payload": {"asset_id": "AGV-EDGE-TEST"}, "ingested_at": time.time()}
    try:
        assert client.get("/api/data-source").json()["edge_active"] >= 1
    finally:
        realtime_server.EDGE_INPUTS.pop("AGV-EDGE-TEST", None)


def test_phm_forecast_endpoint(client):
    r = client.get("/api/phm")
    assert r.status_code == 200
    body = r.json()
    assert body["schema"] == "fab.phm.forecast.v1"
    assert {"current_fault", "predicted_fault", "watch", "normal", "max_risk_score"} <= body["summary"].keys()
    assert body["assets"], "PHM forecast asset list must not be empty"
    first = body["assets"][0]
    assert {"id", "health", "trend_dir", "phm"} <= first.keys()
    one = client.get("/api/phm", params={"agv": first["id"]})
    assert one.status_code == 200
    assert one.json()["assets"][0]["id"] == first["id"]
    metrics = client.get("/metrics").text
    assert "fab_phm_max_risk_score" in metrics
    assert "fab_phm_predicted_fault_assets" in metrics


def test_dispatch_plan_contract(client):
    """PdM 결과가 운영 dispatch/SLA/영향도 계약으로 변환되어야 한다."""
    r = client.get("/api/dispatch-plan")
    assert r.status_code == 200
    body = r.json()
    assert body["schema"] == "fab.dispatch.plan.v1"
    assert {"total", "dispatch_now", "schedule_inspection", "watch", "max_impact_pct"} <= body["summary"].keys()
    assert isinstance(body["assets"], list) and body["assets"]
    first = body["assets"][0]
    assert {"id", "floor", "label", "health", "status", "dispatch"} <= first.keys()
    dispatch = first["dispatch"]
    assert dispatch["state"] in ("dispatch_now", "schedule_inspection", "watch", "normal")
    assert dispatch["priority"] in ("P1", "P2", "P3", "NORMAL")
    assert isinstance(dispatch["affected_zone"], str) and dispatch["affected_zone"]

    one = client.get("/api/dispatch-plan", params={"agv": first["id"]})
    assert one.status_code == 200
    assert one.json()["assets"][0]["id"] == first["id"]

    m = client.get("/metrics").text
    for name in ("fab_ops_dispatch_required_assets", "fab_ops_max_impact_pct"):
        assert f"# TYPE {name} gauge" in m and f"\n{name} " in m


def test_live_booster_inference_contract(client):
    """B1: 3D 트윈 snapshot의 진단은 live LightGBM Booster 경로를 거쳐야 함."""
    s = client.get("/api/snapshot").json()
    a = s["agvs"][0]
    assert a["inference_mode"] == "live_booster"
    assert isinstance(a["model_latency_ms"], (int, float)) and a["model_latency_ms"] >= 0
    assert isinstance(a["replay_pred"], str) and "replay_conf" in a
    assert 0 <= a["conf"] <= 1
    m = client.get("/metrics").text
    for name in ("fab_live_inference_calls", "fab_live_inference_latency_ms"):
        assert f"# TYPE {name} gauge" in m and f"\n{name} " in m


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


def test_trend_rollup_buckets():
    """C4: 시간 버킷 롤업(다운샘플링) — 버킷 경계·평균·등급 집계·시간 오름차순."""
    from telemetry_store import TelemetryStore
    st = TelemetryStore(":memory:")
    ev = lambda h, lv: [{"id": "AGV-01", "floor": 0, "status": "warn", "pred": "E-RBT-B",
                         "conf": 0.9, "level": lv, "health": h,
                         "sensors": {"vib": 5, "batt": 30, "temp": 50}}]
    st.record(100.0, ev(40, "위험")); st.record(110.0, ev(60, "경고"))   # 버킷1 (60초)
    st.record(170.0, ev(80, "주의"))                                     # 버킷2
    tr = st.trend(bucket_sec=60, buckets=10)
    assert len(tr) == 2
    assert tr[0]["t"] < tr[1]["t"], "시간 오름차순이어야 함"
    b1, b2 = tr[0], tr[1]
    assert b1["events"] == 2 and b1["avg_health"] == 50.0
    assert b1["danger"] == 1 and b1["warning"] == 1
    assert b2["events"] == 1 and b2["caution"] == 1


def test_reliability_metrics():
    """C4: 신뢰성 지표(MTBF·MTTR·가용도) — 고장 에피소드 복원·집계."""
    from telemetry_store import TelemetryStore
    st = TelemetryStore(":memory:")
    warn = lambda: [{"id": "AGV-01", "floor": 0, "status": "warn", "pred": "E-RBT-B",
                     "conf": 0.9, "level": "위험", "health": 30,
                     "sensors": {"vib": 5, "batt": 30, "temp": 50}}]
    ok = lambda: [{"id": "AGV-02", "floor": 0, "status": "ok", "pred": "정상",
                   "conf": 0.99, "level": None, "health": 70,   # 저건전도 정상(관측창 확장용)
                   "sensors": {"vib": 2, "batt": 80, "temp": 38}}]
    # AGV-01: 고장 에피소드 2개 (100~101초 연속, 그리고 120초 단발) · 관측창 100~130초
    st.record(100.0, warn()); st.record(100.5, warn()); st.record(101.0, warn())
    st.record(120.0, warn())
    st.record(130.0, ok())
    r1 = st.reliability(agv="AGV-01")
    assert r1["episodes"] == 2, "gap(3초) 초과로 끊긴 에피소드 2개여야 함"
    assert 0 < r1["availability"] < 1 and r1["mttr"] > 0 and r1["mtbf"] > 0
    fleet = st.reliability(n_total=10)
    assert fleet["episodes"] == 2 and 0 < fleet["availability"] <= 1
    assert fleet["worst"] and fleet["worst"][0]["agv"] == "AGV-01"
    # 무고장 설비 포함(n_total=10)이 미포함(1대)보다 가용도가 높아야 함
    assert fleet["availability"] > st.reliability()["availability"] - 1e-9
    # 이벤트 없으면 완전 가용
    empty = TelemetryStore(":memory:").reliability()
    assert empty["availability"] == 1.0 and empty["episodes"] == 0


def test_reliability_and_prometheus_endpoints(client):
    """C2/C4: /api/reliability 계약 + /metrics Prometheus 텍스트 포맷."""
    rel = client.get("/api/reliability").json()
    assert {"window_sec", "episodes", "mttr", "mtbf", "availability", "worst"} <= rel.keys()
    one = client.get("/api/reliability", params={"agv": "AGV-01"}).json()
    assert {"episodes", "mttr", "availability"} <= one.keys()
    m = client.get("/metrics")
    assert m.status_code == 200 and "text/plain" in m.headers["content-type"]
    for name in ("fab_agv_total", "fab_fleet_avg_health", "fab_events_stored",
                 "fab_data_qa_pass_rate", "fab_fleet_availability", "fab_fleet_mttr_seconds"):
        assert f"# TYPE {name} gauge" in m.text and f"\n{name} " in m.text, f"{name} 메트릭 누락"


def test_work_order_endpoint_contract(client):
    """C7: AI 예측 경고를 현장 정비 작업지시(CMMS-style queue)로 전환하는 API 계약."""
    r = client.get("/api/work-orders")
    assert r.status_code == 200
    body = r.json()
    assert {"summary", "orders"} <= body.keys()
    assert {"total", "by_status", "by_priority", "open_by_priority",
            "open_p1", "overdue_open"} <= body["summary"].keys()
    assert isinstance(body["orders"], list)
    if body["orders"]:
        assert {"sla_seconds", "due_ts", "age_sec", "time_to_due_sec", "overdue"} <= body["orders"][0].keys()

    filtered = client.get("/api/work-orders", params={"status": "open", "limit": 5})
    assert filtered.status_code == 200
    assert len(filtered.json()["orders"]) <= 5

    m = client.get("/metrics").text
    for name in ("fab_work_orders_total", "fab_work_orders_open_p1", "fab_work_orders_overdue_open"):
        assert f"# TYPE {name} gauge" in m and f"\n{name} " in m, f"{name} 메트릭 누락"


def test_fleet_risk_endpoint_contract(client):
    """운영 리스크 요약 API: floor별 위험도와 우선 대응 asset 계약."""
    r = client.get("/api/fleet-risk")
    assert r.status_code == 200
    body = r.json()
    assert {"status", "score", "bottleneck_floor", "floor_risk", "top_assets",
            "action_required", "work_orders", "recommendation"} <= body.keys()
    assert body["status"] in ("ok", "watch", "critical")
    assert 0 <= body["score"] <= 100
    assert isinstance(body["floor_risk"], list) and body["floor_risk"]
    assert {"floor", "total", "warn", "avg_health", "score", "critical_assets"} <= body["floor_risk"][0].keys()
    assert isinstance(body["top_assets"], list)

    m = client.get("/metrics").text
    for name in ("fab_fleet_risk_score", "fab_fleet_risk_action_required"):
        assert f"# TYPE {name} gauge" in m and f"\n{name} " in m, f"{name} 메트릭 누락"


def test_ops_report_endpoint_contract(client):
    """운영 리포트 API: risk/work-order/drift/reliability/model 요약 계약."""
    r = client.get("/api/ops-report")
    assert r.status_code == 200
    body = r.json()
    assert body["schema"] == "fab.ops.report.v1"
    assert {"fleet", "risk", "floor_risk", "top_assets", "work_orders",
            "ai_ops", "reliability"} <= body.keys()
    assert body["ai_ops"]["model_id"] == "robot-pdm-lightgbm-enhanced"
    assert body["risk"]["status"] in ("ok", "watch", "critical")

    md = client.get("/api/ops-report", params={"fmt": "md"})
    assert md.status_code == 200
    assert "text/markdown" in md.headers["content-type"]
    assert "# FAB AGV Operations Report" in md.text
    assert "## Top Risk Assets" in md.text


def test_shift_handover_endpoint_contract(client):
    """교대 인수인계 API: checklist/watch asset Markdown export 계약."""
    r = client.get("/api/shift-handover", params={"shift": "night"})
    assert r.status_code == 200
    body = r.json()
    assert body["schema"] == "fab.shift.handover.v1"
    assert body["shift"] == "night"
    assert body["status"] in ("ok", "watch", "critical")
    assert {"summary", "checklist", "watch_assets", "floor_focus", "model"} <= body.keys()
    assert isinstance(body["checklist"], list) and body["checklist"]
    assert {"priority", "category", "action"} <= body["checklist"][0].keys()

    md = client.get("/api/shift-handover", params={"shift": "night", "fmt": "md"})
    assert md.status_code == 200
    assert "text/markdown" in md.headers["content-type"]
    assert "# FAB AGV Shift Handover" in md.text
    assert "## Checklist" in md.text


def test_edge_gateway_endpoint_contract(client):
    """C8: MQTT-style edge telemetry 토픽/스키마/최근 메시지 API 계약."""
    contract = client.get("/api/edge-contract").json()
    assert contract["transport"] == "mqtt-compatible-json"
    assert "{floor}" in contract["topic_pattern"] and "{agv_id}" in contract["topic_pattern"]
    assert "sensors" in contract["payload_required"]

    r = client.get("/api/edge-events", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert {"summary", "events"} <= body.keys()
    assert body["summary"]["schema"] == contract["schema"]
    assert body["events"] and len(body["events"]) <= 5
    event = body["events"][0]
    assert event["topic"].startswith("factory/demo-fab/floor/")
    assert event["validation"]["ok"] is True
    assert {"asset_id", "sensors", "diagnosis", "health", "source"} <= event["payload"].keys()

    m = client.get("/metrics").text
    for name in ("fab_edge_messages_total", "fab_edge_buffered_messages",
                 "fab_edge_active_topics", "fab_edge_invalid_messages",
                 "fab_edge_ingested_messages"):
        assert f"# TYPE {name} gauge" in m and f"\n{name} " in m, f"{name} 메트릭 누락"


def test_edge_ingest_overrides_snapshot_for_mqtt_fed_replay(client):
    """C8: 외부 MQTT-fed payload가 짧은 TTL 동안 3D snapshot/PHM에 반영되어야 함."""
    import time
    from edge_gateway import payload_from_agv
    from realtime_server import EDGE_INPUTS

    base = client.get("/api/snapshot").json()["agvs"][0]
    payload = payload_from_agv(time.time(), base)
    payload["diagnosis"].update({
        "status": "warn",
        "fault": "E-RBT-S",
        "label": "센서 이상",
        "confidence": 0.96,
        "level": "위험",
        "trend": "악화",
    })
    payload["sensors"] = {"vib": 9.9, "batt": 41.0, "temp": 66.0}
    payload["health"] = {"index": 22, "advice": "정비 필요 · 우선 대응"}

    try:
        r = client.post("/api/edge-ingest", json=payload)
        assert r.status_code == 200
        assert r.json()["accepted"] is True

        snap = client.get("/api/snapshot").json()
        asset = next(a for a in snap["agvs"] if a["id"] == base["id"])
        assert asset["edge_input"]["active"] is True
        assert asset["pred"] == "E-RBT-S"
        assert asset["sensors"] == {"vib": 9.9, "batt": 41.0, "temp": 66.0}
        assert asset["health"] == 22
        assert asset["phm"]["stage"] == "current_fault"

        bad = client.post("/api/edge-ingest", json={"schema": "broken"})
        assert bad.status_code == 400
        assert bad.json()["accepted"] is False
    finally:
        EDGE_INPUTS.clear()


def test_trend_endpoint_and_csv_export(client):
    """C4: /api/trend 계약 + /api/history CSV 반출(리포팅 연계)."""
    tr = client.get("/api/trend", params={"bucket": 30, "n": 10}).json()
    assert isinstance(tr, list)
    for b in tr:
        assert {"t", "events", "avg_health", "danger", "warning", "caution"} <= b.keys()
    r = client.get("/api/history", params={"agv": "AGV-01", "fmt": "csv"})
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert r.text.splitlines()[0] == "ts,pred,conf,level,health,vib,batt,temp"


def test_tsdb_export_endpoints(client):
    """C9: 외부 TSDB 확장 계약 — Influx line protocol / Timescale SQL export."""
    contract = client.get("/api/tsdb-contract").json()
    assert contract["schema"] == "fab.telemetry.tsdb_export.v1"
    assert contract["supported_formats"] == ["json", "influx", "timescale"]

    influx = client.get("/api/tsdb-export", params={"fmt": "influx", "limit": 5})
    assert influx.status_code == 200
    assert "text/plain" in influx.headers["content-type"]
    assert "robot_pdm_events" in influx.text

    sql = client.get("/api/tsdb-export", params={"fmt": "timescale", "limit": 5})
    assert sql.status_code == 200
    assert "application/sql" in sql.headers["content-type"]
    assert "CREATE TABLE IF NOT EXISTS robot_pdm_events" in sql.text

    bad = client.get("/api/tsdb-export", params={"fmt": "unknown"})
    assert bad.status_code == 400


def test_data_quality_endpoint(client):
    """로보틱스 학습 데이터셋 QA/거버넌스 지표 계약."""
    q = client.get("/api/data-quality").json()
    for key in ("total", "schema_valid_rate", "annotation_coverage", "qa_pass_rate",
                "ingest_success_rate", "rework_rate", "by_modality", "issues"):
        assert key in q
    assert q["total"] > 0
    assert 0 <= q["schema_valid_rate"] <= 1
    assert 0 <= q["qa_pass_rate"] <= 1


def test_data_drift_endpoint_and_metrics(client):
    """운영 AI 모니터링: 실시간 입력 분포 드리프트 API + Prometheus 게이지 계약."""
    d = client.get("/api/drift").json()
    for key in ("status", "score", "window_size", "features", "fault_rate",
                "drifted_features", "watch_features", "recommendation"):
        assert key in d
    assert d["status"] in ("ok", "watch", "drift")
    assert d["window_size"] > 0
    assert {"vib", "batt", "temp", "health", "conf"} <= d["features"].keys()
    assert 0 <= d["fault_rate"]["current"] <= 1
    m = client.get("/metrics")
    for name in ("fab_data_drift_score", "fab_data_drift_features", "fab_data_drift_fault_rate"):
        assert f"# TYPE {name} gauge" in m.text and f"\n{name} " in m.text, f"{name} 메트릭 누락"


def test_model_card_endpoint(client):
    """모델 거버넌스: 관제 서버에서도 artifact hash와 피처 계약을 조회할 수 있어야 함."""
    card = client.get("/api/model-card").json()
    assert card["model_id"] == "robot-pdm-lightgbm-enhanced"
    assert card["performance"]["official_validation"]["split"].startswith("AI-Hub official")
    assert card["feature_engineering"]["contract_ok"] is True
    assert card["input_contract"]["excluded_dynamic_sensors"] == ["x", "y"]


def test_reviewer_brief_endpoint(client):
    """포트폴리오 리뷰어가 3분 안에 볼 경로와 직무 매핑을 API로 제공."""
    brief = client.get("/api/reviewer-brief").json()
    assert brief["review_time_minutes"] == 3
    assert brief["start_here"][0]["path"] == "/twin"
    assert any(point["claim"] == "MLOps and governance are represented" for point in brief["proof_points"])
    assert any(role["role"] == "Robotics / Smart Factory Engineer" for role in brief["role_mapping"])
