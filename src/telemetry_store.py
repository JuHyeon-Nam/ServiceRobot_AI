"""
telemetry_store.py — 실시간 진단 이벤트 시계열 데이터 계층 (SQLite, 무설치)

관제 서버가 매 틱 산출하는 AGV 진단을 '흘려보내고 끝'이 아니라,
시계열로 **적재(ingest) → 롤업 집계(aggregate) → 이력 조회(query) → 보존정책(retention)**
하는 경량 데이터 파이프라인. 표준 라이브러리 sqlite3만 사용(추가 설치 없음).

- 기본 인메모리(:memory:) → 서버 세션 동안 누적, 재시작 시 리셋(데모 친화적)
- 환경변수 TELEMETRY_DB 에 파일 경로를 주면 durable 저장
- 저장량은 '이상/저건전도 이벤트만 선별 적재' + max_rows 보존정책으로 바운드

→ 향후 MQTT 수집·외부 시계열DB(InfluxDB/TimescaleDB) 교체 시에도 동일 인터페이스로 확장 가능.
"""
import sqlite3
import threading
from collections import defaultdict

_DDL = """CREATE TABLE IF NOT EXISTS events(
  ts     REAL,      -- epoch seconds
  agv    TEXT,
  floor  INTEGER,
  pred   TEXT,
  conf   REAL,
  level  TEXT,      -- 주의/경고/위험 (정상 이벤트는 적재 안 함)
  health INTEGER,   -- 0~100 자산 건전도
  vib    REAL, batt REAL, temp REAL
)"""


class TelemetryStore:
    """진단 이벤트 시계열 스토어. 스레드 안전(단일 커넥션 + 락)."""

    def __init__(self, path: str = ":memory:", max_rows: int = 20000):
        # check_same_thread=False: uvicorn의 워커 스레드/이벤트루프에서 공용 접근 허용
        self.cx = sqlite3.connect(path, check_same_thread=False)
        self.cx.execute(_DDL)
        self.cx.execute("CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts)")
        self.cx.execute("CREATE INDEX IF NOT EXISTS ix_events_agv ON events(agv, ts)")
        self.cx.commit()
        self.lock = threading.Lock()
        self.max_rows = max_rows

    def record(self, ts: float, agvs: list) -> int:
        """스트리밍 적재. 이상(warn) 또는 저건전도(health<80) 이벤트만 선별 저장.
        정상·안정은 스킵 → 저장량을 의미 있는 이벤트로 바운드."""
        rows = []
        for a in agvs:
            if a.get("status") == "warn" or a.get("health", 100) < 80:
                s = a.get("sensors") or {}
                rows.append((ts, a["id"], a["floor"], a["pred"], a["conf"],
                             a.get("level"), a.get("health", 100),
                             s.get("vib"), s.get("batt"), s.get("temp")))
        if not rows:
            return 0
        with self.lock:
            self.cx.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
            self.cx.commit()
        return len(rows)

    def prune(self) -> int:
        """보존정책: 최신 max_rows 행만 유지, 오래된 이벤트 삭제. 삭제 건수 반환."""
        with self.lock:
            cur = self.cx.execute(
                "DELETE FROM events WHERE rowid NOT IN "
                "(SELECT rowid FROM events ORDER BY ts DESC LIMIT ?)", (self.max_rows,))
            self.cx.commit()
            return cur.rowcount

    def history(self, agv: str, limit: int = 200) -> list:
        """설비 1대의 최근 진단 이벤트 이력(최신순)."""
        with self.lock:
            cur = self.cx.execute(
                "SELECT ts,pred,conf,level,health,vib,batt,temp FROM events "
                "WHERE agv=? ORDER BY ts DESC LIMIT ?", (agv, limit))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def events(self, limit: int = 1000, since: float = 0.0) -> list:
        """외부 TSDB export용 최근 이벤트 목록(최신순)."""
        with self.lock:
            cur = self.cx.execute(
                "SELECT ts,agv,floor,pred,conf,level,health,vib,batt,temp FROM events "
                "WHERE ts>=? ORDER BY ts DESC LIMIT ?", (since, limit))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def trend(self, bucket_sec: int = 60, buckets: int = 15) -> list:
        """시간 버킷 롤업(다운샘플링): 버킷별 이벤트 수·평균 건전도·등급별 집계.
        시계열DB의 continuous aggregate 개념을 경량 구현 — 관제 추이 차트의 데이터 소스.
        반환은 시간 오름차순(차트가 왼→오른쪽으로 그리도록)."""
        with self.lock:
            rows = self.cx.execute(
                "SELECT CAST(ts/? AS INTEGER) b, COUNT(*), AVG(health), "
                "SUM(level='위험'), SUM(level='경고'), SUM(level='주의') "
                "FROM events GROUP BY b ORDER BY b DESC LIMIT ?",
                (bucket_sec, buckets)).fetchall()
        out = [{"t": r[0] * bucket_sec, "events": r[1],
                "avg_health": round(r[2], 1) if r[2] is not None else None,
                "danger": r[3], "warning": r[4], "caution": r[5]} for r in rows]
        return list(reversed(out))

    def reliability(self, agv: str = None, gap: float = 3.0, n_total: int = None) -> dict:
        """신뢰성 지표(MTBF·MTTR·가용도) — 이벤트 스트림에서 '고장 에피소드'를 복원해 계산.
        warn 이벤트(level 있음)가 gap초 이내로 연속되면 하나의 고장 구간(에피소드)으로 묶는다.
        - MTTR  = 총 다운타임 / 에피소드 수 (평균 복구 시간)
        - MTBF  = 총 가동시간 / 에피소드 수 (평균 고장 간격)
        - 가용도 = 가동시간 / 관측시간  (신뢰성 공학의 Availability)
        n_total(전체 설비 수)을 주면 무고장 설비까지 포함한 플릿 가용도를 계산한다."""
        q = "SELECT agv, ts FROM events WHERE level IS NOT NULL"
        args: tuple = ()
        if agv:
            q += " AND agv=?"; args = (agv,)
        q += " ORDER BY agv, ts"
        with self.lock:
            rows = self.cx.execute(q, args).fetchall()
            span = self.cx.execute("SELECT MIN(ts), MAX(ts) FROM events").fetchone()
        zero = {"episodes": 0, "mttr": 0.0, "mtbf": None, "availability": 1.0}
        if span[0] is None:
            return dict(zero, window_sec=0.0) if agv else dict(zero, window_sec=0.0, worst=[])
        window = max(span[1] - span[0], 1e-9)
        per = defaultdict(list)
        for a, ts in rows:
            per[a].append(ts)
        SAMPLE = 0.5                       # 적재 주기(2Hz) 보정: 단일 샘플도 최소 이 길이의 고장으로 간주
        per_out = {}
        for a, tss in per.items():
            eps, down, start, prev = 0, 0.0, tss[0], tss[0]
            for t in tss[1:]:
                if t - prev > gap:         # 간격이 벌어지면 에피소드 종료
                    eps += 1; down += (prev - start) + SAMPLE
                    start = t
                prev = t
            eps += 1; down += (prev - start) + SAMPLE
            down = min(down, window)
            up = max(window - down, 0.0)
            per_out[a] = {"episodes": eps, "mttr": round(down / eps, 1),
                          "mtbf": round(up / eps, 1), "availability": round(up / window, 4)}
        if agv:
            return dict(per_out.get(agv, zero), window_sec=round(window, 1))
        # 플릿 집계: 무고장 설비 포함(n_total) — 총 가동시간 대비 총 다운타임
        n = max(n_total or len(per_out), len(per_out), 1)
        total_down = sum(window - v["availability"] * window for v in per_out.values())
        total_eps = sum(v["episodes"] for v in per_out.values())
        fleet_up = window * n - total_down
        worst = sorted(per_out.items(), key=lambda kv: kv[1]["availability"])[:5]
        return {"window_sec": round(window, 1), "episodes": total_eps,
                "mttr": round(total_down / max(total_eps, 1), 1),
                "mtbf": round(fleet_up / max(total_eps, 1), 1),
                "availability": round(fleet_up / (window * n), 4),
                "worst": [dict(agv=a, **v) for a, v in worst]}

    def stats(self, since: float = 0.0) -> dict:
        """세션 누적 롤업 집계: 총 이벤트·등급별·층별 분포·최다 결함 AGV·평균 건전도."""
        with self.lock:
            total = self.cx.execute("SELECT COUNT(*) FROM events WHERE ts>=?", (since,)).fetchone()[0]
            by_level = dict(self.cx.execute(
                "SELECT level,COUNT(*) FROM events WHERE ts>=? AND level IS NOT NULL "
                "GROUP BY level", (since,)).fetchall())
            by_floor = dict(self.cx.execute(
                "SELECT floor,COUNT(*) FROM events WHERE ts>=? GROUP BY floor", (since,)).fetchall())
            top = self.cx.execute(
                "SELECT agv,COUNT(*) c FROM events WHERE ts>=? GROUP BY agv "
                "ORDER BY c DESC LIMIT 5", (since,)).fetchall()
            avg_h = self.cx.execute("SELECT AVG(health) FROM events WHERE ts>=?", (since,)).fetchone()[0]
        return {"total": total, "by_level": by_level,
                "by_floor": {str(k): v for k, v in by_floor.items()},
                "top_agv": [{"agv": a, "events": c} for a, c in top],
                "avg_health": round(avg_h, 1) if avg_h is not None else None}
