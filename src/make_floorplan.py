"""
make_floorplan.py — 반도체 FAB AGV 예지보전 관제 시각화
  - assets/control_center.png : 팹 라인(장비 베이 + AGV 트랙) 정적 뷰
  - assets/control_center.gif : AGV가 트랙을 따라 이동하며 AI 진단색이 바뀌는 애니메이션
AGV 이동 = 각 트랙 구간에 매핑된 '실제 좌표 궤적'(겹치지 않게 구간 분할), 색 = 모델의 실제 예측.
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
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter

for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False

DATA, ASSETS = "../data/processed", "../assets"
TOOL_FC, TOOL_EC, TLAB = "#eef1f5", "#aab2bf", "#5b6472"
AISLE, RAIL = "#f1f3f6", "#c9ced6"
GREEN, GREEN_E, RED, RED_E, FOUP = "#16a34a", "#15803d", "#dc2626", "#b91c1c", "#475569"
KOR = {"E-ENV-C": "교차로 충돌위험", "E-ENV-O": "경로 장애물", "E-INF-A": "도어 연동",
       "E-INF-E": "층간리프트 연동", "E-RBT-B": "배터리 저하", "E-RBT-E": "긴급정지",
       "E-RBT-N": "통신 끊김", "E-RBT-S": "센서 이상", "정상": "정상"}

# 장비 베이(공정 툴) — 3행 x 6열
TOOL_ROWS = [
    (82, ["PHOTO", "ETCH", "CVD", "CMP", "DIFF", "IMP"]),
    (56, ["CLEAN", "METRO", "DRY-ETCH", "PVD", "ANNEAL", "CDS"]),
    (30, ["WET", "LITHO", "EPI", "IMP-2", "INSP", "COAT"]),
]
TCX = [22, 46, 70, 94, 118, 142]      # 툴 x중심
AISLE_Y = [73, 47, 21]                 # 베이 사이 AGV 트랙
MAIN_Y = 11                            # 하부 메인 트랙
TRACK_X = (10, 150)

# AGV → 트랙 구간(겹치지 않게 분할): (트랙y, x0, x1)
def segs(y, n, x0=10, x1=150):
    e = np.linspace(x0, x1, n + 1)
    return [(y, e[i] + 2, e[i + 1] - 2) for i in range(n)]
AGV_SEGS = segs(AISLE_Y[0], 3) + segs(AISLE_Y[1], 3) + segs(AISLE_Y[2], 2) + \
           segs(MAIN_Y, 2, 10, 192)


def draw_fab(ax):
    ax.set_facecolor("#fafafa")
    # 외곽(클린룸 경계)
    ax.add_patch(Rectangle((6, 5), 188, 92, fc="none", ec="#374151", lw=2.0, zorder=2))
    ax.text(8, 99.5, "FAB BAY · 1", fontsize=7, color="#9ca3af", va="bottom", zorder=2)
    # AGV 트랙(밴드 + 점선 중심선 + 레일틱)
    for ay in AISLE_Y + [MAIN_Y]:
        x0, x1 = (10, 192) if ay == MAIN_Y else TRACK_X
        ax.add_patch(Rectangle((x0, ay - 2.6), x1 - x0, 5.2, fc=AISLE, ec="none", zorder=1.4))
        ax.plot([x0, x1], [ay, ay], color=RAIL, lw=0.9, ls=(0, (5, 4)), zorder=1.5)
        for xx in np.arange(x0, x1, 6):
            ax.plot([xx, xx], [ay - 2.6, ay - 1.8], color=RAIL, lw=0.7, zorder=1.5)
            ax.plot([xx, xx], [ay + 1.8, ay + 2.6], color=RAIL, lw=0.7, zorder=1.5)
    # 수직 연결 트랙(스토커로)
    ax.add_patch(Rectangle((148, MAIN_Y - 2.6), 5.2, 64, fc=AISLE, ec="none", zorder=1.3))
    # 공정 장비 베이
    for ry, labels in TOOL_ROWS:
        for cx, lab in zip(TCX, labels):
            ax.add_patch(FancyBboxPatch((cx - 10, ry), 20, 13, boxstyle="round,pad=0,rounding_size=0.8",
                         fc=TOOL_FC, ec=TOOL_EC, lw=1.1, zorder=2.2))
            ax.text(cx, ry + 9.5, lab, fontsize=6.8, color=TLAB, ha="center", va="center",
                    weight="bold", zorder=2.3)
            ax.text(cx, ry + 5, "EQUIP", fontsize=4.6, color="#aab2bf", ha="center", zorder=2.3)
            # 로드포트(하단 작은 칸)
            for k in range(3):
                ax.add_patch(Rectangle((cx - 7.5 + k * 5, ry - 0.2), 3.4, 1.6,
                             fc="#dfe3ea", ec=TOOL_EC, lw=0.5, zorder=2.3))
    # 스토커(WIP 보관)
    ax.add_patch(FancyBboxPatch((158, 18), 30, 70, boxstyle="round,pad=0,rounding_size=1",
                 fc="#e8edf3", ec=TOOL_EC, lw=1.2, zorder=2.2))
    ax.text(173, 84, "STOCKER", fontsize=8, color=TLAB, ha="center", weight="bold", zorder=2.3)
    ax.text(173, 80, "(WIP 보관)", fontsize=6, color="#9ca3af", ha="center", zorder=2.3)
    for r in range(5):
        for c in range(4):
            ax.add_patch(Rectangle((161 + c * 6.6, 24 + r * 10), 5.2, 7.2,
                         fc="#f4f7fb", ec=TOOL_EC, lw=0.5, zorder=2.3))
    ax.set_xlim(2, 198); ax.set_ylim(0, 104)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def agv(ax, x, y, right, col, ec):
    """AGV 카트(본체 + FOUP + 진행 노즈)"""
    ax.add_patch(FancyBboxPatch((x - 3, y - 1.7), 6, 3.4, boxstyle="round,pad=0,rounding_size=0.6",
                 fc=col, ec=ec, lw=1.0, alpha=0.95, zorder=5))
    ax.add_patch(Rectangle((x - 1.3, y + 0.4), 2.6, 2.2, fc=FOUP, ec="none", zorder=5.2))  # FOUP
    d = 1 if right else -1
    ax.add_patch(Polygon([(x + d * 3, y - 1.4), (x + d * 4.4, y), (x + d * 3, y + 1.4)],
                 closed=True, fc=col, ec=ec, lw=0.8, zorder=5))


def main():
    rep = pd.read_parquet(f"{DATA}/replay.parquet")
    robots = sorted(rep.robot.unique())[:len(AGV_SEGS)]
    names = {rid: f"AGV-{i+1:02d}" for i, rid in enumerate(robots)}
    R = {}
    for rid, (ay, x0, x1) in zip(robots, AGV_SEGS):
        g = rep[rep.robot == rid].sort_values("seq").reset_index(drop=True)
        px = g.px.to_numpy(); nx = (px - px.min()) / (np.ptp(px) + 1e-9)
        X = x0 + nx * (x1 - x0)
        R[rid] = dict(x=X, y=ay, pred=g["pred"].to_numpy(), n=len(g))

    def fi(rid, p):
        return int(p * (R[rid]["n"] - 1))

    def render(ax, p):
        draw_fab(ax)
        nwarn = 0
        for rid in robots:
            d = R[rid]; i = fi(rid, p)
            err = d["pred"][i] != "정상"; col = RED if err else GREEN; ec = RED_E if err else GREEN_E
            nwarn += err
            x, y = d["x"][i], d["y"]
            right = (d["x"][i] - d["x"][max(0, i - 1)]) >= 0
            # 트레일(트랙 위)
            a = max(0, i - 10); xs = d["x"][a:i + 1]
            for k in range(1, len(xs)):
                ax.plot([xs[k - 1], xs[k]], [y, y], color=col,
                        alpha=0.06 + 0.3 * k / len(xs), lw=2.2, zorder=4, solid_capstyle="round")
            if err:
                ax.add_patch(Circle((x, y), 4.2, color=RED, alpha=0.13, zorder=4.5))
            agv(ax, x, y, right, col, ec)
            ax.text(x, y + 3.4, names[rid], fontsize=5.6, color="#374151", ha="center", zorder=6)
            if err:
                ax.text(x, y - 4.2, KOR.get(d["pred"][i], d["pred"][i]), fontsize=5.4,
                        color=RED, ha="center", zorder=6)
        # 헤더 + 상태
        ax.text(6, 102.5, "FAB AMHS MONITOR", fontsize=12.5, weight="bold", color="#111", va="center")
        ax.text(74, 102.5, "반도체 라인 AGV 실시간 예지보전 관제", fontsize=8.5, color="#666", va="center")
        ax.text(194, 102.5, f"AGV {len(robots)}   정상 {len(robots)-nwarn}   경고 {nwarn}",
                fontsize=9.5, weight="bold", ha="right",
                color=(RED if nwarn else "#15803d"), va="center")
        ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=GREEN, label="정상 AGV", markersize=9),
                           Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, label="고장 경고", markersize=9)],
                  loc="lower left", fontsize=7.5, frameon=False, ncol=2)
        return nwarn

    best = 0.5
    for p in np.linspace(0.1, 0.95, 40):
        if 2 <= sum(R[r]["pred"][fi(r, p)] != "정상" for r in robots) <= 3:
            best = p; break
    fig, ax = plt.subplots(figsize=(13.5, 7.3)); fig.patch.set_facecolor("#fafafa")
    render(ax, best); fig.tight_layout()
    fig.savefig(f"{ASSETS}/control_center.png", dpi=135, facecolor="#fafafa"); plt.close(fig)
    print("저장: control_center.png")

    figg, axg = plt.subplots(figsize=(13.5, 7.3)); figg.patch.set_facecolor("#fafafa")
    NF, ps = 64, np.linspace(0, 1, 64)
    def upd(k):
        axg.clear(); render(axg, ps[k]); return []
    FuncAnimation(figg, upd, frames=NF, interval=130).save(
        f"{ASSETS}/control_center.gif", writer=PillowWriter(fps=8), dpi=80)
    plt.close(figg)
    print(f"저장: control_center.gif ({os.path.getsize(f'{ASSETS}/control_center.gif')/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
