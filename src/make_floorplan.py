"""
make_floorplan.py — 반도체 FAB(다층) AGV 예지보전 관제 시각화
  - assets/control_center.png : 3개 층(2F/1F/B1) 팹 + AMHS 트랙 루프 정적 뷰
  - assets/control_center.gif : AGV 15대가 트랙 루프를 따라 '코너를 꺾으며' 순환하고
                               AI 진단색이 바뀌는 애니메이션
경로 = 장비 베이를 도는 직각 트랙 루프(코너 포함), 진행 속도는 로봇의 실제 궤적 거리로 변조,
색 = 모델의 실제 예측. (데이터/모델은 서비스 로봇 공개데이터 → 팹 AGV PdM 컨셉)
실행: cd src && python make_floorplan.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter

for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False

DATA, ASSETS = "../data/processed", "../assets"
PAGE, PANEL, PHEAD, PEC = "#f6f8fb", "#ffffff", "#eef2f7", "#d4dae3"
TOOL_FC, TOOL_EC, TLAB = "#e9edf3", "#b6bfcb", "#5b6472"
TRKBAND, RAIL = "#e9edf3", "#b9c2d0"
GREEN, GREEN_E, RED, RED_E, FOUP = "#16a34a", "#15803d", "#dc2626", "#b91c1c", "#3f4b5e"
HEAD_BG = "#0f172a"
KOR = {"E-ENV-C": "교차로 충돌위험", "E-ENV-O": "경로 장애물", "E-INF-A": "도어 연동",
       "E-INF-E": "층간리프트 연동", "E-RBT-B": "배터리 저하", "E-RBT-E": "긴급정지",
       "E-RBT-N": "통신 끊김", "E-RBT-S": "센서 이상", "정상": "정상"}

FLOORS = [
    (94, "2F · 포토 / 식각 베이", ["PHOTO", "SCANNER", "TRACK", "DRY-ETCH", "ASH", "CD-SEM"]),
    (50, "1F · 박막 / 확산 베이", ["CVD", "PVD", "ALD", "CMP", "DIFF", "ANNEAL"]),
    (6,  "B1 · 서브팹 / 유틸리티", ["PUMP", "SCRUBBER", "CHILLER", "GAS-BOX", "UPW", "FFU"]),
]
FH = 40
PX0, PX1 = 6, 184
TCX = [24, 50, 76, 102, 128, 154]


def floor_geo(y0):
    xL, xR, xM = 15, 173, 94
    yB, yT = y0 + 7, y0 + 30
    return xL, xR, xM, yB, yT


class Route:
    """직각 폴리라인(닫힌 루프) 위를 호 길이로 따라가는 경로."""
    def __init__(self, pts):
        self.p = np.array(pts, float)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(self.p, axis=0).T))]
        self.cum, self.total = d, d[-1]

    def at(self, s):
        d = (s % 1.0) * self.total
        k = np.searchsorted(self.cum, d) - 1
        k = np.clip(k, 0, len(self.p) - 2)
        seg = self.cum[k + 1] - self.cum[k] + 1e-9
        f = (d - self.cum[k]) / seg
        a, b = self.p[k], self.p[k + 1]
        xy = a + (b - a) * f
        ang = np.arctan2(b[1] - a[1], b[0] - a[0])
        return xy[0], xy[1], ang


def routes_for(y0):
    xL, xR, xM, yB, yT = floor_geo(y0)
    outer = Route([(xL, yB), (xL, yT), (xR, yT), (xR, yB), (xL, yB)])
    fig8 = Route([(xL, yB), (xM, yB), (xM, yT), (xR, yT), (xR, yB),
                  (xM, yB), (xM, yT), (xL, yT), (xL, yB)])
    return outer, fig8


def rot(pts, ang, cx, cy):
    c, s = np.cos(ang), np.sin(ang)
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in pts]


CART = [(-2.6, -1.3), (1.3, -1.3), (3.1, 0), (1.3, 1.3), (-2.6, 1.3)]
FBOX = [(-1.6, -1.0), (0.3, -1.0), (0.3, 1.0), (-1.6, 1.0)]


def draw_agv(ax, x, y, ang, col, ec):
    ax.add_patch(Polygon(rot(CART, ang, x, y), closed=True, fc=col, ec=ec, lw=0.9, alpha=0.97, zorder=5))
    ax.add_patch(Polygon(rot(FBOX, ang, x, y), closed=True, fc=FOUP, ec="none", zorder=5.2))


def trk(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=TRKBAND, lw=4.6, solid_capstyle="round", zorder=1.5)
    ax.plot([x0, x1], [y0, y1], color=RAIL, lw=0.9, ls=(0, (5, 4)), zorder=1.6)


def draw_fab(ax):
    ax.set_facecolor(PAGE)
    ax.add_patch(Rectangle((0, 138), 232, 12, fc=HEAD_BG, ec="none", zorder=8))
    for y0, name, labels in FLOORS:
        xL, xR, xM, yB, yT = floor_geo(y0)
        ax.add_patch(FancyBboxPatch((PX0, y0), PX1 - PX0, FH, boxstyle="round,pad=0,rounding_size=1.2",
                     fc=PANEL, ec=PEC, lw=1.3, zorder=1.2))
        ax.add_patch(Rectangle((PX0 + 0.6, y0 + FH - 6), PX1 - PX0 - 1.2, 5.4, fc=PHEAD, ec="none", zorder=1.3))
        ax.text(PX0 + 3, y0 + FH - 3.3, name, fontsize=8, color="#374151", va="center", weight="bold", zorder=1.4)
        # AMHS 트랙 루프(직각 + 중앙 연결로) — AGV가 이 경로를 따라 코너를 꺾음
        trk(ax, xL, yB, xR, yB); trk(ax, xL, yT, xR, yT)
        trk(ax, xL, yB, xL, yT); trk(ax, xR, yB, xR, yT); trk(ax, xM, yB, xM, yT)
        for cx, lab in zip(TCX, labels):   # 장비 베이(루프 안쪽)
            ax.add_patch(FancyBboxPatch((cx - 10.5, y0 + 13.5), 21, 11, boxstyle="round,pad=0,rounding_size=0.7",
                         fc=TOOL_FC, ec=TOOL_EC, lw=1.0, zorder=2.4))
            ax.text(cx, y0 + 20.5, lab, fontsize=6.4, color=TLAB, ha="center", va="center", weight="bold", zorder=2.5)
            ax.text(cx, y0 + 16.5, "EQUIP", fontsize=4.2, color="#aeb6c2", ha="center", zorder=2.5)
    # 스토커 + 리프트
    sx0 = 188
    ax.add_patch(FancyBboxPatch((sx0, 6), 22, 128, boxstyle="round,pad=0,rounding_size=1.2",
                 fc="#e7ecf3", ec=PEC, lw=1.3, zorder=1.2))
    ax.text(sx0 + 11, 130, "STOCKER", fontsize=8, color=TLAB, ha="center", weight="bold", zorder=1.4)
    ax.text(sx0 + 11, 126, "WIP 보관", fontsize=5.5, color="#9ca3af", ha="center", zorder=1.4)
    for r in range(11):
        for c in range(3):
            ax.add_patch(Rectangle((sx0 + 2.5 + c * 6, 10 + r * 10.3), 5, 7.6,
                         fc="#f3f6fa", ec=TOOL_EC, lw=0.45, zorder=1.4))
    lx = 214
    ax.add_patch(FancyBboxPatch((lx, 6), 14, 128, boxstyle="round,pad=0,rounding_size=1.2",
                 fc="#dfe6ef", ec=PEC, lw=1.2, zorder=1.2))
    ax.text(lx + 7, 130, "LIFT", fontsize=7.5, color=TLAB, ha="center", weight="bold", zorder=1.4)
    for y0, *_ in FLOORS:
        ax.add_patch(Rectangle((lx + 3, y0 + 8), 8, 16, fc="#eef3f9", ec=TOOL_EC, lw=0.8, zorder=1.4))
        ax.annotate("", xy=(lx + 7, y0 + 26), xytext=(lx + 7, y0 + 6),
                    arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.2), zorder=1.5)
    ax.set_xlim(0, 232); ax.set_ylim(0, 150)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    rep = pd.read_parquet(f"{DATA}/replay.parquet")
    robots = sorted(rep.robot.unique())
    traj = {}
    for rid in robots:
        g = rep[rep.robot == rid].sort_values("seq").reset_index(drop=True)
        px, py = g.px.to_numpy(), g.py.to_numpy()
        step = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
        traj[rid] = dict(prog=step / (step[-1] + 1e-9), pred=g["pred"].to_numpy(), n=len(g))

    # 층마다 5대: outer 3 + fig8 2, 위상차로 분산
    AGV = {}
    idx = 0
    for fidx, (y0, *_rest) in enumerate(FLOORS):
        outer, fig8 = routes_for(y0)
        plan = [(outer, 0.0), (outer, 0.34), (outer, 0.67), (fig8, 0.12), (fig8, 0.62)]
        for route, ph in plan:
            AGV[f"AGV-{idx+1:02d}"] = dict(route=route, ph=ph, rid=robots[idx % len(robots)], floor=fidx)
            idx += 1

    def sample(a, p):
        t = traj[a["rid"]]; i = int(p * (t["n"] - 1))
        s = (t["prog"][i] + a["ph"]) % 1.0
        x, y, ang = a["route"].at(s)
        return x, y, ang, s, t["pred"][i]

    def render(ax, p):
        draw_fab(ax)
        per = [0, 0, 0]; tot = 0
        for name, a in AGV.items():
            x, y, ang, s, pred = sample(a, p)
            err = pred != "정상"; col = RED if err else GREEN; ec = RED_E if err else GREEN_E
            per[a["floor"]] += err; tot += err
            # 경로를 따라가는 트레일(코너 포함)
            for b in range(8, 0, -1):
                xb, yb, _ = a["route"].at(s - b * 0.01)
                ax.add_patch(Circle((xb, yb), 0.5, color=col, alpha=0.05 + 0.30 * (8 - b) / 8, zorder=4))
            if err:
                ax.add_patch(Circle((x, y), 3.4, color=RED, alpha=0.14, zorder=4.5))
            draw_agv(ax, x, y, ang, col, ec)
            ax.text(x, y + 3.0, name, fontsize=4.7, color="#374151", ha="center", zorder=6)
            if err:
                ax.text(x, y - 3.4, KOR.get(pred, pred), fontsize=4.5, color=RED, ha="center", zorder=6)
        ax.text(4, 144, "FAB AMHS MONITOR", fontsize=12.5, weight="bold", color="white", va="center", zorder=9)
        ax.text(86, 144, "반도체 라인(3F) AGV 실시간 예지보전 관제", fontsize=8.5, color="#94a3b8", va="center", zorder=9)
        n = len(AGV)
        ax.text(228, 144, f"AGV {n}   정상 {n-tot}   경고 {tot}", fontsize=10, weight="bold",
                ha="right", color=("#fca5a5" if tot else "#86efac"), va="center", zorder=9)
        for (y0, *_), w in zip(FLOORS, per):
            ax.text(PX1 - 3, y0 + FH - 3.3, f"경고 {w}", fontsize=6.5, ha="right", va="center",
                    color=(RED if w else "#15803d"), zorder=1.5)
        ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=GREEN, label="정상 AGV", markersize=9),
                           Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, label="고장 경고", markersize=9)],
                  loc="lower left", bbox_to_anchor=(0.005, -0.02), fontsize=7.5, frameon=False, ncol=2)
        return tot

    best = 0.5
    for p in np.linspace(0.05, 0.95, 50):
        if 2 <= sum(sample(a, p)[4] != "정상" for a in AGV.values()) <= 4:
            best = p; break
    fig, ax = plt.subplots(figsize=(15.5, 9.8)); fig.patch.set_facecolor(PAGE)
    render(ax, best); fig.tight_layout()
    fig.savefig(f"{ASSETS}/control_center.png", dpi=130, facecolor=PAGE); plt.close(fig)
    print("저장: control_center.png")

    figg, axg = plt.subplots(figsize=(15.5, 9.8)); figg.patch.set_facecolor(PAGE)
    NF, ps = 60, np.linspace(0, 1, 60)
    def upd(k):
        axg.clear(); render(axg, ps[k]); return []
    FuncAnimation(figg, upd, frames=NF, interval=110).save(
        f"{ASSETS}/control_center.gif", writer=PillowWriter(fps=10), dpi=72)
    plt.close(figg)
    print(f"저장: control_center.gif ({os.path.getsize(f'{ASSETS}/control_center.gif')/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
