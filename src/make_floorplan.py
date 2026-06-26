"""
make_floorplan.py — 반도체 FAB(다층) AGV 예지보전 관제 시각화
  - assets/control_center.png : 3개 층(2F/1F/B1) 팹 라인 + AGV 트랙 정적 뷰
  - assets/control_center.gif : AGV 15대가 트랙을 따라 이동하며 AI 진단색이 바뀌는 애니메이션
AGV 이동 = 트랙 구간에 매핑된 '실제 좌표 궤적'(겹치지 않게 분할 + 위상차), 색 = 모델의 실제 예측.
* 데이터/모델은 서비스 로봇 공개데이터이며, 고장 유형이 팹 AGV PdM에 대응되어 그 컨셉으로 표현.
실행: cd src && python make_floorplan.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle, FancyArrow
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter

for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False

DATA, ASSETS = "../data/processed", "../assets"
PAGE, PANEL, PHEAD, PEC = "#f6f8fb", "#ffffff", "#eef2f7", "#d4dae3"
TOOL_FC, TOOL_EC, TLAB = "#e9edf3", "#b6bfcb", "#5b6472"
AISLE, RAIL = "#eef1f5", "#c8ced8"
GREEN, GREEN_E, RED, RED_E, FOUP = "#16a34a", "#15803d", "#dc2626", "#b91c1c", "#475569"
HEAD_BG, HEAD_TX = "#0f172a", "#e2e8f0"
KOR = {"E-ENV-C": "교차로 충돌위험", "E-ENV-O": "경로 장애물", "E-INF-A": "도어 연동",
       "E-INF-E": "층간리프트 연동", "E-RBT-B": "배터리 저하", "E-RBT-E": "긴급정지",
       "E-RBT-N": "통신 끊김", "E-RBT-S": "센서 이상", "정상": "정상"}

# 층 정의: (y0, 이름, 툴 라벨 6개)  — 위에서 아래로 2F→1F→B1
FLOORS = [
    (94, "2F · 포토 / 식각 베이", ["PHOTO", "SCANNER", "TRACK", "DRY-ETCH", "ASH", "CD-SEM"]),
    (50, "1F · 박막 / 확산 베이", ["CVD", "PVD", "ALD", "CMP", "DIFF", "ANNEAL"]),
    (6,  "B1 · 서브팹 / 유틸리티", ["PUMP", "SCRUBBER", "CHILLER", "GAS-BOX", "UPW", "FFU"]),
]
FH = 40                      # 층 패널 높이
PX0, PX1 = 6, 184            # 팹 패널 x범위
TCX = [22, 50, 78, 106, 134, 162]    # 툴 x중심
TRX = (12, 180)              # 트랙 x범위


def tool(ax, cx, ty, lab):
    ax.add_patch(FancyBboxPatch((cx - 11, ty), 22, 11, boxstyle="round,pad=0,rounding_size=0.7",
                 fc=TOOL_FC, ec=TOOL_EC, lw=1.0, zorder=2.4))
    ax.text(cx, ty + 7.4, lab, fontsize=6.6, color=TLAB, ha="center", va="center", weight="bold", zorder=2.5)
    ax.text(cx, ty + 3.2, "EQUIP", fontsize=4.3, color="#aeb6c2", ha="center", zorder=2.5)
    for k in range(3):       # 로드포트
        ax.add_patch(Rectangle((cx - 8 + k * 5.3, ty - 0.2), 3.6, 1.5, fc="#dde2ea", ec=TOOL_EC, lw=0.4, zorder=2.5))


def track(ax, y, x0=TRX[0], x1=TRX[1]):
    ax.add_patch(Rectangle((x0, y - 2.4), x1 - x0, 4.8, fc=AISLE, ec="none", zorder=1.6))
    ax.plot([x0, x1], [y, y], color=RAIL, lw=0.9, ls=(0, (5, 4)), zorder=1.7)
    for xx in np.arange(x0, x1, 6):
        ax.plot([xx, xx], [y - 2.4, y - 1.7], color=RAIL, lw=0.6, zorder=1.7)
        ax.plot([xx, xx], [y + 1.7, y + 2.4], color=RAIL, lw=0.6, zorder=1.7)


def agv(ax, x, y, right, col, ec):
    ax.add_patch(FancyBboxPatch((x - 2.7, y - 1.5), 5.4, 3.0, boxstyle="round,pad=0,rounding_size=0.5",
                 fc=col, ec=ec, lw=0.9, alpha=0.96, zorder=5))
    ax.add_patch(Rectangle((x - 1.1, y + 0.35), 2.2, 1.9, fc=FOUP, ec="none", zorder=5.2))
    d = 1 if right else -1
    ax.add_patch(Polygon([(x + d * 2.7, y - 1.2), (x + d * 4, y), (x + d * 2.7, y + 1.2)],
                 closed=True, fc=col, ec=ec, lw=0.7, zorder=5))


def draw_fab(ax):
    ax.set_facecolor(PAGE)
    # 상단 헤더바
    ax.add_patch(Rectangle((0, 138), 232, 10, fc=HEAD_BG, ec="none", zorder=8))
    # 층 패널 + 장비 + 트랙
    for y0, name, labels in FLOORS:
        ax.add_patch(FancyBboxPatch((PX0, y0), PX1 - PX0, FH, boxstyle="round,pad=0,rounding_size=1.2",
                     fc=PANEL, ec=PEC, lw=1.3, zorder=1.2))
        ax.add_patch(Rectangle((PX0 + 0.6, y0 + FH - 6), PX1 - PX0 - 1.2, 5.4, fc=PHEAD, ec="none", zorder=1.3))
        ax.text(PX0 + 3, y0 + FH - 3.3, name, fontsize=8, color="#374151", va="center", weight="bold", zorder=1.4)
        for cx, lab in zip(TCX, labels):       # 툴 1행
            tool(ax, cx, y0 + 19, lab)
        track(ax, y0 + 13)                      # 트랙 A
        track(ax, y0 + 4.5)                     # 트랙 B
    # 스토커 타워(전 층 관통)
    sx0 = 188
    ax.add_patch(FancyBboxPatch((sx0, 6), 22, 128, boxstyle="round,pad=0,rounding_size=1.2",
                 fc="#e7ecf3", ec=PEC, lw=1.3, zorder=1.2))
    ax.text(sx0 + 11, 130, "STOCKER", fontsize=8, color=TLAB, ha="center", weight="bold", zorder=1.4)
    ax.text(sx0 + 11, 126, "WIP 보관", fontsize=5.5, color="#9ca3af", ha="center", zorder=1.4)
    for r in range(11):
        for c in range(3):
            ax.add_patch(Rectangle((sx0 + 2.5 + c * 6, 10 + r * 10.3), 5, 7.6,
                         fc="#f3f6fa", ec=TOOL_EC, lw=0.45, zorder=1.4))
    # 층간 리프트(우측), 화살표로 층 연결
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
        px = g.px.to_numpy(); nx = (px - px.min()) / (np.ptp(px) + 1e-9)
        traj[rid] = dict(nx=nx, pred=g["pred"].to_numpy(), n=len(g))

    # 트랙 구성: 층마다 A/B 2트랙, 트랙마다 2~3 AGV  → 총 15대
    plan = []  # (track_y, x0, x1, robot, phase, floor_idx)
    counts = [3, 2, 3, 2, 3, 2]
    ri = 0
    for fidx, (y0, *_rest) in enumerate(FLOORS):
        for tj, ty in enumerate([y0 + 13, y0 + 4.5]):
            cnt = counts[fidx * 2 + tj]
            e = np.linspace(TRX[0] + 4, TRX[1] - 4, cnt + 1)
            for j in range(cnt):
                rid = robots[ri % len(robots)]
                phase = 0.0 if ri < len(robots) else 0.45
                plan.append((ty, e[j] + 3, e[j + 1] - 3, rid, phase, fidx))
                ri += 1
    AGV = {f"AGV-{i+1:02d}": p for i, p in enumerate(plan)}

    def state(name, p):
        ty, x0, x1, rid, ph, fidx = AGV[name]
        t = traj[rid]; i = int(((p + ph) % 1.0) * (t["n"] - 1))
        x = x0 + t["nx"][i] * (x1 - x0)
        xprev = x0 + t["nx"][max(0, i - 1)] * (x1 - x0)
        return x, ty, x >= xprev, t["pred"][i], fidx, x0, x1

    def render(ax, p):
        draw_fab(ax)
        per = [0, 0, 0]; tot_w = 0
        for name in AGV:
            x, y, right, pred, fidx, x0, x1 = state(name, p)
            err = pred != "정상"; col = RED if err else GREEN; ec = RED_E if err else GREEN_E
            per[fidx] += err; tot_w += err
            # 트레일
            ty, _x0, _x1, rid, ph, _ = AGV[name]; t = traj[rid]
            i = int(((p + ph) % 1.0) * (t["n"] - 1)); a = max(0, i - 8)
            xs = x0 + t["nx"][a:i + 1] * (x1 - x0)
            for k in range(1, len(xs)):
                ax.plot([xs[k - 1], xs[k]], [y, y], color=col, alpha=0.05 + 0.28 * k / len(xs),
                        lw=2.0, zorder=4, solid_capstyle="round")
            if err:
                ax.add_patch(Circle((x, y), 3.6, color=RED, alpha=0.13, zorder=4.5))
            agv(ax, x, y, right, col, ec)
            ax.text(x, y + 2.9, name, fontsize=4.8, color="#374151", ha="center", zorder=6)
            if err:
                ax.text(x, y - 3.6, KOR.get(pred, pred), fontsize=4.6, color=RED, ha="center", zorder=6)
        # 헤더 텍스트 + 글로벌 KPI
        ax.text(4, 143, "FAB AMHS MONITOR", fontsize=12.5, weight="bold", color="white", va="center", zorder=9)
        ax.text(86, 143, "반도체 라인(3F) AGV 실시간 예지보전 관제", fontsize=8.5, color="#94a3b8", va="center", zorder=9)
        n = len(AGV)
        ax.text(228, 143, f"AGV {n}   정상 {n-tot_w}   경고 {tot_w}", fontsize=10, weight="bold",
                ha="right", color=("#fca5a5" if tot_w else "#86efac"), va="center", zorder=9)
        # 층별 KPI(패널 헤더 우측)
        for (y0, *_), w in zip(FLOORS, per):
            ax.text(PX1 - 3, y0 + FH - 3.3, f"경고 {w}", fontsize=6.5, ha="right", va="center",
                    color=(RED if w else "#15803d"), zorder=1.5)
        ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=GREEN, label="정상 AGV", markersize=9),
                           Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, label="고장 경고", markersize=9)],
                  loc="lower left", bbox_to_anchor=(0.005, -0.02), fontsize=7.5, frameon=False, ncol=2)
        return tot_w

    best = 0.5
    for p in np.linspace(0.05, 0.95, 50):
        if 2 <= sum(state(nm, p)[3] != "정상" for nm in AGV) <= 4:
            best = p; break
    fig, ax = plt.subplots(figsize=(15.5, 9.8)); fig.patch.set_facecolor(PAGE)
    render(ax, best); fig.tight_layout()
    fig.savefig(f"{ASSETS}/control_center.png", dpi=130, facecolor=PAGE); plt.close(fig)
    print("저장: control_center.png")

    figg, axg = plt.subplots(figsize=(15.5, 9.8)); figg.patch.set_facecolor(PAGE)
    NF, ps = 56, np.linspace(0, 1, 56)
    def upd(k):
        axg.clear(); render(axg, ps[k]); return []
    FuncAnimation(figg, upd, frames=NF, interval=140).save(
        f"{ASSETS}/control_center.gif", writer=PillowWriter(fps=8), dpi=72)
    plt.close(figg)
    print(f"저장: control_center.gif ({os.path.getsize(f'{ASSETS}/control_center.gif')/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
