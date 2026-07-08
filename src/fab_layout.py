"""
fab_layout.py — 대규모 팹(다층) 도면·AMHS 트랙의 단일 소스(서버/프론트/시각화 공유)
- 좌표계: matplotlib식 y-up, 캔버스 300 x 196 (프론트에서 y 반전)
- 층마다 베이-메시(가로 3 아일 × 세로 5 커넥터) + 장비 16대 + OHT 레일,
  AGV는 메시 위 다양한 직각 루프를 따라 코너를 꺾으며 순환.
"""
import numpy as np

CANVAS = {"w": 300, "h": 196}
FH = 60
PX0, PX1 = 8, 252                       # 팹 영역 x
VX = [16, 74, 132, 190, 248]            # 세로 메시(커넥터) x
# 층: (y0, 약칭이름, 풀네임, 장비 약어 풀)
FLOORS_DEF = [
    (128, "2F", "포토 / 식각", ["PHO", "SCN", "TRK", "ETC", "ASH", "DEV", "CMP", "CDS"]),
    (66,  "1F", "박막 / 확산", ["CVD", "PVD", "ALD", "EPI", "DIF", "RTP", "ANN", "CLN"]),
    (4,   "B1", "서브팹 / 유틸", ["PMP", "SCB", "CHL", "GAS", "UPW", "FFU", "N2", "EXH"]),
]


def fgeo(y0):
    return dict(yA=y0 + 8, yB=y0 + 26, yC=y0 + 45, y0=y0)


class Route:
    def __init__(self, pts):
        self.p = np.array(pts, float)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(self.p, axis=0).T))]
        self.cum, self.total = d, d[-1]

    def at(self, s):
        d = (s % 1.0) * self.total
        k = int(np.clip(np.searchsorted(self.cum, d) - 1, 0, len(self.p) - 2))
        seg = self.cum[k + 1] - self.cum[k] + 1e-9
        f = (d - self.cum[k]) / seg
        a, b = self.p[k], self.p[k + 1]
        xy = a + (b - a) * f
        ang = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
        return float(xy[0]), float(xy[1]), ang


def loop(x0, x1, y0, y1):
    return [(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]


def _floor_tracks(g):
    yA, yC = g["yA"], g["yC"]; yB = g["yB"]
    h = [[VX[0], y, VX[-1], y] for y in (yA, yB, yC)]
    v = [[vx, yA, vx, yC] for vx in VX]
    return h + v


def _floor_tools(g, pool):
    yA, yB, yC = g["yA"], g["yB"], g["yC"]
    tools, n = [], 0
    for i in range(len(VX) - 1):                       # 4 cells
        x0, x1 = VX[i], VX[i + 1]; cw = x1 - x0
        for (lo, hi) in ((yA, yB), (yB, yC)):          # 2 bands
            cy = (lo + hi) / 2; th = (hi - lo) * 0.62
            for s in (-1, 1):                          # 2 tools/cell-band
                cx = x0 + cw * (0.5 + s * 0.24); tw = cw * 0.40
                n += 1
                tools.append({"cx": round(cx, 1), "cy": round(cy, 1),
                              "w": round(tw, 1), "h": round(th, 1),
                              "label": f"{pool[(n-1) % len(pool)]}-{n:02d}"})
    return tools


def build_layout():
    floors = []
    for y0, short, full, pool in FLOORS_DEF:
        g = fgeo(y0)
        floors.append({
            "y0": y0, "short": short, "name": f"{short} · {full} 베이",
            "geo": g, "tracks": _floor_tracks(g), "tools": _floor_tools(g, pool),
            "oht": [[VX[0], g["yC"] + 6, VX[-1], g["yC"] + 6]],   # 오버헤드 OHT 레일
        })
    stocker = {"x": 256, "y": 6, "w": 24, "h": 184, "cols": 3, "rows": 16}
    lift = {"x": 282, "y": 6, "w": 14, "h": 184,
            "cabs": [{"y": y0 + 10, "h": 20} for y0, *_ in FLOORS_DEF]}
    return {"canvas": CANVAS, "floors": floors, "stocker": stocker, "lift": lift}


def build_agv_plan(robots):
    """층마다 9대(셀루프4 + 더블루프2x2 + 페리미터1). 코너 많은 다양한 루프."""
    plan, idx = [], 0
    for fidx, (y0, *_rest) in enumerate(FLOORS_DEF):
        g = fgeo(y0); yA, yC = g["yA"], g["yC"]
        routes = []
        for i in range(4):                              # 셀 루프 4
            routes.append((Route(loop(VX[i], VX[i + 1], yA, yC)), 0.0))
        routes.append((Route(loop(VX[0], VX[2], yA, yC)), 0.0))   # 더블 좌
        routes.append((Route(loop(VX[0], VX[2], yA, yC)), 0.5))
        routes.append((Route(loop(VX[2], VX[4], yA, yC)), 0.0))   # 더블 우
        routes.append((Route(loop(VX[2], VX[4], yA, yC)), 0.5))
        routes.append((Route(loop(VX[0], VX[4], yA, yC)), 0.2))   # 페리미터
        for route, ph in routes:
            # toff: AGV마다 재생 타임라인 오프셋(황금비 분산) → 고장이 한꺼번에 몰리지 않고 현실적으로 흩어짐
            plan.append({"id": f"AGV-{idx+1:02d}", "route": route, "phase": ph,
                         "robot": robots[idx % len(robots)], "floor": fidx,
                         "toff": round((idx * 0.61803398875) % 1.0, 4)})
            idx += 1
    return plan
