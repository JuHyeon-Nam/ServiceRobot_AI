"""
make_floorplan.py — 건축 도면(CAD) 기반 관제 시각화
  - assets/control_center.png : CAD 평면도 + 로봇(실제 좌표 이동) 정적 뷰
  - assets/control_center.gif : 로봇이 실제 궤적대로 이동하며 AI 진단색이 바뀌는 애니메이션
로봇 위치 = 각 구역에 배치된 '실제 좌표 궤적'(겹치지 않게 구역 분할), 색 = 모델의 실제 예측.
실행: cd src && python make_floorplan.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter

for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False

DATA, ASSETS = "../data/processed", "../assets"
WALL, GRID, DIM, COL, FIX, RLAB = "#2b2b2b", "#dcdcdc", "#888", "#1a1a1a", "#555", "#9aa3af"
GREEN, RED = "#15a34a", "#dc2626"
KOR = {"E-ENV-C": "혼잡·충돌위험", "E-ENV-O": "장애물", "E-INF-A": "자동문연동",
       "E-INF-E": "엘리베이터연동", "E-RBT-B": "배터리저하", "E-RBT-E": "긴급정지",
       "E-RBT-N": "네트워크끊김", "E-RBT-S": "센서이상", "정상": "정상"}

# 건물 외곽 / 그리드 -------------------------------------------------
BX0, BY0, BX1, BY1 = 12, 10, 168, 104
GX = [40, 66, 92, 118, 144]            # 세로 그리드(기둥열)
GY = [33.5, 57, 80.5]                  # 가로 그리드  → 4밴드(=4x8,700)

# 로봇 배치 구역(x0,y0,w,h,label) — 서로 겹치지 않음 (코어 영역은 제외)
ZONES = [
    (14, 82, 48, 20, "사무실 A"), (14, 59, 48, 21, "회의실"),
    (14, 12, 48, 19, "로비 · 입구"), (66, 12, 24, 24, "복도"),
    (94, 80, 36, 22, "오픈 오피스 A"), (132, 80, 34, 22, "사무 구역"),
    (94, 55, 34, 22, "오픈 오피스 B"), (132, 55, 34, 22, "오픈 오피스 C"),
    (94, 12, 36, 26, "라운지"), (132, 12, 34, 26, "홀"),
]


def draw_cad(ax):
    ax.set_facecolor("white")
    # 그리드(점선) + 버블
    for i, gx in enumerate(GX):
        ax.plot([gx, gx], [BY0 - 4, BY1 + 6], color=GRID, lw=0.8, ls=(0, (6, 4)), zorder=1)
        ax.add_patch(Circle((gx, BY1 + 9), 2.6, fc="white", ec="#777", lw=0.9, zorder=6))
        ax.text(gx, BY1 + 9, str(i + 1), ha="center", va="center", fontsize=7, color="#555", zorder=7)
    for j, gy in enumerate(GY):
        ax.plot([BX0 - 6, BX1 + 4], [gy, gy], color=GRID, lw=0.8, ls=(0, (6, 4)), zorder=1)
        ax.add_patch(Circle((BX0 - 11, gy), 2.6, fc="white", ec="#777", lw=0.9, zorder=6))
        ax.text(BX0 - 11, gy, chr(65 + j), ha="center", va="center", fontsize=7, color="#555", zorder=7)
    # 좌측 치수선 (8,700 x4 = 34,800)
    edges = [BY0] + GY + [BY1]
    dimx = BX0 - 17
    ax.plot([dimx, dimx], [BY0, BY1], color=DIM, lw=0.8, zorder=2)
    for a, b in zip(edges[:-1], edges[1:]):
        for yy in (a, b):
            ax.plot([dimx - 1.4, dimx + 1.4], [yy, yy], color=DIM, lw=0.8, zorder=2)
        ax.text(dimx - 2.5, (a + b) / 2, "8,700", rotation=90, ha="center", va="center",
                fontsize=6, color=DIM, zorder=2)
    ax.text(dimx - 8, (BY0 + BY1) / 2, "34,800", rotation=90, ha="center", va="center",
            fontsize=7.5, color="#333", weight="bold", zorder=2)
    # 외벽(이중선 느낌)
    ax.add_patch(Rectangle((BX0, BY0), BX1 - BX0, BY1 - BY0, fc="none", ec=WALL, lw=2.4, zorder=3))
    ax.add_patch(Rectangle((BX0 + 1.1, BY0 + 1.1), BX1 - BX0 - 2.2, BY1 - BY0 - 2.2,
                 fc="none", ec=WALL, lw=0.6, zorder=3))
    # 기둥(그리드 교차점 채운 사각)
    for gx in GX:
        for gy in [BY0] + GY + [BY1]:
            ax.add_patch(Rectangle((gx - 1.1, gy - 1.1), 2.2, 2.2, fc=COL, ec="none", zorder=4))
    # 실 구획(벽)
    for x0, y0, w, h, label in ZONES:
        if w <= 0:
            continue
        ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=WALL, lw=1.1, zorder=3))
        if not label.startswith("_"):
            ax.text(x0 + 1.6, y0 + h - 2, label, fontsize=7.5, color=RLAB, va="top", zorder=3.5)
    # 코어(중앙 컬럼 x66~90, y38~102): 엘리베이터2 + 계단 + 화장실
    ax.add_patch(Rectangle((66, 38, ), 24, 64, fc="#f7f7f7", ec=WALL, lw=1.1, zorder=2.9))
    ax.text(67.5, 100.5, "코어", fontsize=7, color=RLAB, va="top", zorder=3.7)
    for ey in (86, 78):  # 엘리베이터(대각 X)
        ax.add_patch(Rectangle((68, ey), 9, 7, fc="none", ec=FIX, lw=1.0, zorder=3.6))
        ax.plot([68, 77], [ey, ey + 7], color=FIX, lw=0.7, zorder=3.6)
        ax.plot([68, 77], [ey + 7, ey], color=FIX, lw=0.7, zorder=3.6)
    ax.text(72.5, 94.5, "EV", fontsize=6, color=FIX, ha="center", zorder=3.7)
    sx, sy = 79, 78  # 계단
    ax.add_patch(Rectangle((sx, sy), 9, 15, fc="none", ec=FIX, lw=1.0, zorder=3.6))
    for k in range(1, 8):
        ax.plot([sx, sx + 9], [sy + k * 15 / 8, sy + k * 15 / 8], color=FIX, lw=0.5, zorder=3.6)
    ax.text(sx + 4.5, sy + 16.5, "STAIR", fontsize=6, color=FIX, ha="center", zorder=3.7)
    ax.add_patch(Rectangle((68, 42), 20, 14, fc="none", ec=WALL, lw=1.0, zorder=3.6))  # 화장실
    for k in range(1, 5):
        ax.plot([68 + k * 20 / 5, 68 + k * 20 / 5], [49, 56], color=FIX, lw=0.6, zorder=3.6)
    ax.text(69, 43.4, "화장실", fontsize=6.5, color=RLAB, va="bottom", zorder=3.7)
    ax.set_xlim(BX0 - 24, BX1 + 8); ax.set_ylim(BY0 - 8, BY1 + 16)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    rep = pd.read_parquet(f"{DATA}/replay.parquet")
    usable = [z for z in ZONES if z[2] > 0]
    robots = sorted(rep.robot.unique())[:len(usable)]
    R = {}
    for rid, (x0, y0, w, h, _) in zip(robots, usable):
        g = rep[rep.robot == rid].sort_values("seq").reset_index(drop=True)
        px, py = g.px.to_numpy(), g.py.to_numpy()
        nx = (px - px.min()) / (np.ptp(px) + 1e-9); ny = (py - py.min()) / (np.ptp(py) + 1e-9)
        pad = 0.16
        R[rid] = dict(x=x0 + w * (pad + nx * (1 - 2 * pad)),
                      y=y0 + h * (pad + ny * (1 - 2 * pad)),
                      deg=g.degree.to_numpy(), pred=g["pred"].to_numpy(), n=len(g))

    def fi(rid, p):
        return int(p * (R[rid]["n"] - 1))

    def render(ax, p):
        draw_cad(ax)
        nwarn = 0
        for rid in robots:
            d = R[rid]; i = fi(rid, p)
            err = d["pred"][i] != "정상"; col = RED if err else GREEN; nwarn += err
            a = max(0, i - 12)
            xs, ys = d["x"][a:i + 1], d["y"][a:i + 1]
            for k in range(1, len(xs)):
                ax.plot(xs[k - 1:k + 1], ys[k - 1:k + 1], color=col,
                        alpha=0.06 + 0.32 * k / len(xs), lw=1.3, zorder=5, solid_capstyle="round")
            x, y = d["x"][i], d["y"][i]
            ax.add_patch(Circle((x, y), 2.6, color=col, alpha=0.16, zorder=5.4))
            if err:
                ax.add_patch(Circle((x, y), 2.0, fc="none", ec=col, lw=1.4, zorder=5.6))
            ax.add_patch(Circle((x, y), 1.25, color=col, ec="white", lw=0.7, zorder=6))
            th = np.deg2rad(d["deg"][i] if not np.isnan(d["deg"][i]) else 0)
            ax.plot([x, x + 2.6 * np.cos(th)], [y, y + 2.6 * np.sin(th)], color=col, lw=1.2, zorder=6)
            ax.text(x, y + 3.4, rid, color="#374151", fontsize=6, ha="center", zorder=6.2)
            if err:
                ax.text(x, y - 3.8, KOR.get(d["pred"][i], d["pred"][i]), color=RED,
                        fontsize=5.6, ha="center", zorder=6.2)
        # 제목 + 상태
        ax.text(BX0 - 24, BY1 + 13, "ROBOT CONTROL CENTER",
                fontsize=12, weight="bold", color="#111", va="center")
        ax.text(BX0 - 24, BY1 + 8.5, "서비스 로봇 실시간 예지보전 관제  ·  1F 평면도",
                fontsize=8, color="#666", va="center")
        ax.text(BX1 + 6, BY1 + 13, f"가동 {len(robots)}   정상 {len(robots)-nwarn}   경고 {nwarn}",
                fontsize=9.5, weight="bold", ha="right",
                color=(RED if nwarn else "#15803d"), va="center")
        ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, label="정상", markersize=8),
                           Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, label="고장 경고", markersize=8)],
                  loc="lower right", fontsize=7, frameon=False)
        return nwarn

    # 정적 히어로
    best = 0.5
    for p in np.linspace(0.1, 0.95, 40):
        if 2 <= sum(R[r]["pred"][fi(r, p)] != "정상" for r in robots) <= 3:
            best = p; break
    fig, ax = plt.subplots(figsize=(13, 8)); fig.patch.set_facecolor("white")
    render(ax, best); fig.tight_layout()
    fig.savefig(f"{ASSETS}/control_center.png", dpi=135, facecolor="white"); plt.close(fig)
    print("저장: control_center.png")

    # GIF
    figg, axg = plt.subplots(figsize=(13, 8)); figg.patch.set_facecolor("white")
    NF, ps = 64, np.linspace(0, 1, 64)
    def upd(k):
        axg.clear(); render(axg, ps[k]); return []
    FuncAnimation(figg, upd, frames=NF, interval=130).save(
        f"{ASSETS}/control_center.gif", writer=PillowWriter(fps=8), dpi=80)
    plt.close(figg)
    print(f"저장: control_center.gif ({os.path.getsize(f'{ASSETS}/control_center.gif')/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
