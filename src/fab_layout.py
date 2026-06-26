"""
fab_layout.py — 팹(다층) 도면·AMHS 경로의 단일 소스(서버/프론트 공유)
- 좌표계: matplotlib식 y-up, 캔버스 232 x 150 (프론트에서 y 반전)
- 층/장비/트랙 루프/AGV 배치를 정의하고 JSON-가능한 layout과 AGV plan을 제공.
"""
import numpy as np

CANVAS = {"w": 232, "h": 150}
FH = 40
PX0, PX1 = 6, 184
TCX = [24, 50, 76, 102, 128, 154]
FLOORS_DEF = [
    (94, "2F · 포토 / 식각 베이", ["PHOTO", "SCANNER", "TRACK", "DRY-ETCH", "ASH", "CD-SEM"]),
    (50, "1F · 박막 / 확산 베이", ["CVD", "PVD", "ALD", "CMP", "DIFF", "ANNEAL"]),
    (6,  "B1 · 서브팹 / 유틸리티", ["PUMP", "SCRUBBER", "CHILLER", "GAS-BOX", "UPW", "FFU"]),
]


def floor_geo(y0):
    return dict(xL=15, xR=173, xM=94, yB=y0 + 7, yT=y0 + 30)


def routes_for(y0):
    g = floor_geo(y0); xL, xR, xM, yB, yT = g["xL"], g["xR"], g["xM"], g["yB"], g["yT"]
    outer = [(xL, yB), (xL, yT), (xR, yT), (xR, yB), (xL, yB)]
    fig8 = [(xL, yB), (xM, yB), (xM, yT), (xR, yT), (xR, yB), (xM, yB), (xM, yT), (xL, yT), (xL, yB)]
    return outer, fig8


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


def build_layout():
    floors = []
    for y0, name, labels in FLOORS_DEF:
        g = floor_geo(y0); xL, xR, xM, yB, yT = g["xL"], g["xR"], g["xM"], g["yB"], g["yT"]
        tracks = [[xL, yB, xR, yB], [xL, yT, xR, yT], [xL, yB, xL, yT],
                  [xR, yB, xR, yT], [xM, yB, xM, yT]]
        tools = [{"cx": cx, "y": y0 + 13.5, "w": 21, "h": 11, "label": lab}
                 for cx, lab in zip(TCX, labels)]
        floors.append({"y0": y0, "name": name, "geo": g, "tracks": tracks, "tools": tools})
    stocker = {"x": 188, "y": 6, "w": 22, "h": 128, "cols": 3, "rows": 11}
    lift = {"x": 214, "y": 6, "w": 14, "h": 128,
            "cabs": [{"y": y0 + 8, "h": 16} for y0, *_ in FLOORS_DEF]}
    return {"canvas": CANVAS, "floors": floors, "stocker": stocker, "lift": lift}


def build_agv_plan(robots):
    """층마다 5대(outer 3 + fig8 2). 반환: [{id, route(Route), phase, robot, floor}]"""
    plan, idx = [], 0
    for fidx, (y0, *_rest) in enumerate(FLOORS_DEF):
        outer_pts, fig8_pts = routes_for(y0)
        outer, fig8 = Route(outer_pts), Route(fig8_pts)
        for route, ph in [(outer, 0.0), (outer, 0.34), (outer, 0.67), (fig8, 0.12), (fig8, 0.62)]:
            plan.append({"id": f"AGV-{idx+1:02d}", "route": route, "phase": ph,
                         "robot": robots[idx % len(robots)], "floor": fidx})
            idx += 1
    return plan
