"""
make_floorplan.py — 대규모 반도체 FAB(다층) AGV 관제 시각화 (fab_layout 기반)
  - assets/control_center.png : 3층 팹(베이-메시 트랙, 장비 48, AGV 27) 정적 뷰
  - assets/control_center.gif : AGV가 메시 루프를 코너 꺾으며 순환하는 애니메이션
경로/도면은 fab_layout.py 단일 소스(라이브 대시보드와 동일). 색=모델 실제 예측.
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
import fab_layout as FL

for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False

DATA, ASSETS = "../data/processed", "../assets"
PAGE, PANEL, PHEAD, PEC = "#f6f8fb", "#ffffff", "#e9eef5", "#d4dae3"
TOOL_FC, TOOL_EC, TLAB = "#e9edf3", "#b6bfcb", "#5b6472"
TRKBAND, RAIL, OHT = "#e6ebf2", "#aab6c9", "#7dd3fc"
GREEN, GREEN_E, RED, RED_E, FOUP = "#16a34a", "#15803d", "#dc2626", "#b91c1c", "#3f4b5e"
HEAD_BG = "#0f172a"
KOR = {"E-ENV-C": "교차로 충돌위험", "E-ENV-O": "경로 장애물", "E-INF-A": "도어 연동",
       "E-INF-E": "층간리프트 연동", "E-RBT-B": "배터리 저하", "E-RBT-E": "긴급정지",
       "E-RBT-N": "통신 끊김", "E-RBT-S": "센서 이상", "정상": "정상"}
W, H = FL.CANVAS["w"], FL.CANVAS["h"]
CART = [(-2.4, -1.2), (1.2, -1.2), (2.9, 0), (1.2, 1.2), (-2.4, 1.2)]
FBOX = [(-1.5, -0.9), (0.3, -0.9), (0.3, 0.9), (-1.5, 0.9)]


def rot(pts, ang, cx, cy):
    c, s = np.cos(ang), np.sin(ang)
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in pts]


def draw_fab(ax, layout):
    ax.set_facecolor(PAGE)
    ax.add_patch(Rectangle((0, H - 10), W + 4, 14, fc=HEAD_BG, ec="none", zorder=8))
    for f in layout["floors"]:
        y0 = f["y0"]; ax.add_patch(FancyBboxPatch((FL.PX0 - 2, y0 + 2), FL.PX1 - FL.PX0 + 4, FL.FH - 4,
                     boxstyle="round,pad=0,rounding_size=1.4", fc=PANEL, ec=PEC, lw=1.3, zorder=1.1))
        g = f["geo"]
        ax.add_patch(Rectangle((FL.PX0 - 1, g["y0"] + FL.FH - 8), FL.PX1 - FL.PX0 + 2, 5.4, fc=PHEAD, ec="none", zorder=1.2))
        ax.text(FL.PX0, g["y0"] + FL.FH - 5.3, f["name"], fontsize=8.5, color="#374151", va="center", weight="bold", zorder=1.3)
        # OHT 레일(오버헤드)
        for o in f["oht"]:
            ax.plot([o[0], o[2]], [o[1], o[3]], color=OHT, lw=1.6, alpha=0.8, zorder=1.25)
            ax.plot([o[0], o[2]], [o[1] + 0.7, o[3] + 0.7], color=OHT, lw=0.7, alpha=0.5, zorder=1.25)
        ax.text(FL.VX[-1] - 1, f["oht"][0][1] + 1.4, "OHT", fontsize=5, color="#38bdf8", ha="right", zorder=1.3)
        # 메시 트랙
        for t in f["tracks"]:
            ax.plot([t[0], t[2]], [t[1], t[3]], color=TRKBAND, lw=4.4, solid_capstyle="round", zorder=1.5)
            ax.plot([t[0], t[2]], [t[1], t[3]], color=RAIL, lw=0.8, ls=(0, (5, 4)), zorder=1.6)
        # 교차점(노드)
        for vx in FL.VX:
            for vy in (g["yA"], g["yB"], g["yC"]):
                ax.add_patch(Circle((vx, vy), 0.8, fc="#cdd6e4", ec="none", zorder=1.7))
        # 장비
        for tl in f["tools"]:
            ax.add_patch(FancyBboxPatch((tl["cx"] - tl["w"] / 2, tl["cy"] - tl["h"] / 2), tl["w"], tl["h"],
                         boxstyle="round,pad=0,rounding_size=0.5", fc=TOOL_FC, ec=TOOL_EC, lw=0.9, zorder=2.2))
            ax.text(tl["cx"], tl["cy"], tl["label"], fontsize=4.6, color=TLAB, ha="center", va="center", weight="bold", zorder=2.3)
    s = layout["stocker"]
    ax.add_patch(FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"], boxstyle="round,pad=0,rounding_size=1.2",
                 fc="#e7ecf3", ec=PEC, lw=1.3, zorder=1.2))
    ax.text(s["x"] + s["w"] / 2, s["y"] + s["h"] - 4, "STOCKER", fontsize=7.5, color=TLAB, ha="center", weight="bold", zorder=1.4)
    for r in range(s["rows"]):
        for c in range(s["cols"]):
            ax.add_patch(Rectangle((s["x"] + 2.5 + c * 6.6, s["y"] + 4 + r * 11), 5.4, 8, fc="#f3f6fa", ec=TOOL_EC, lw=0.4, zorder=1.4))
    l = layout["lift"]
    ax.add_patch(FancyBboxPatch((l["x"], l["y"]), l["w"], l["h"], boxstyle="round,pad=0,rounding_size=1.2",
                 fc="#dfe6ef", ec=PEC, lw=1.2, zorder=1.2))
    ax.text(l["x"] + l["w"] / 2, l["y"] + l["h"] - 4, "LIFT", fontsize=7, color=TLAB, ha="center", weight="bold", zorder=1.4)
    for cab in l["cabs"]:
        ax.add_patch(Rectangle((l["x"] + 3, cab["y"]), 8, cab["h"], fc="#eef3f9", ec=TOOL_EC, lw=0.7, zorder=1.4))
        ax.annotate("", xy=(l["x"] + 7, cab["y"] + cab["h"]), xytext=(l["x"] + 7, cab["y"]),
                    arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.0), zorder=1.5)
    ax.set_xlim(-2, W + 4); ax.set_ylim(-2, H + 6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_agv(ax, x, y, ang, col, ec):
    ax.add_patch(Polygon(rot(CART, ang, x, y), closed=True, fc=col, ec=ec, lw=0.8, alpha=0.97, zorder=5))
    ax.add_patch(Polygon(rot(FBOX, ang, x, y), closed=True, fc=FOUP, ec="none", zorder=5.2))


def main():
    rep = pd.read_parquet(f"{DATA}/replay.parquet")
    robots = sorted(rep["robot"].unique())
    traj = {}
    for rid in robots:
        gg = rep[rep.robot == rid].sort_values("seq").reset_index(drop=True)
        px, py = gg.px.to_numpy(), gg.py.to_numpy()
        step = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
        traj[rid] = dict(prog=step / (step[-1] + 1e-9), pred=gg["pred"].to_numpy(), n=len(gg))
    layout = FL.build_layout(); plan = FL.build_agv_plan(robots)

    def sample(a, p):
        t = traj[a["robot"]]; i = int(p * (t["n"] - 1))
        s = (t["prog"][i] + a["phase"]) % 1.0
        x, y, ang = a["route"].at(s)
        return x, y, np.deg2rad(ang), s, t["pred"][i]

    def render(ax, p):
        draw_fab(ax, layout)
        per = [0, 0, 0]; tot = 0
        for a in plan:
            x, y, ang, s, pred = sample(a, p)
            err = pred != "정상"; col = RED if err else GREEN; ec = RED_E if err else GREEN_E
            per[a["floor"]] += int(err); tot += int(err)
            for b in range(6, 0, -1):
                xb, yb, _ = a["route"].at(s - b * 0.012)
                ax.add_patch(Circle((xb, yb), 0.4, color=col, alpha=0.05 + 0.28 * (6 - b) / 6, zorder=4))
            if err:
                ax.add_patch(Circle((x, y), 3.0, color=RED, alpha=0.15, zorder=4.5))
            draw_agv(ax, x, y, ang, col, ec)
            if err:
                ax.text(x, y - 3.0, KOR.get(pred, pred), fontsize=4.2, color=RED, ha="center", zorder=6)
        ax.text(3, H - 4.5, "FAB AMHS MONITOR", fontsize=12.5, weight="bold", color="white", va="center", zorder=9)
        ax.text(108, H - 4.5, "반도체 라인(3F) AGV 실시간 예지보전 관제", fontsize=8, color="#94a3b8", va="center", zorder=9)
        n = len(plan)
        ax.text(W + 2, H - 4.5, f"AGV {n}   정상 {n-tot}   경고 {tot}", fontsize=9.5, weight="bold",
                ha="right", color=("#fca5a5" if tot else "#86efac"), va="center", zorder=9)
        for f, w in zip(layout["floors"], per):
            ax.text(FL.PX1, f["geo"]["y0"] + FL.FH - 5.3, f"경고 {w}", fontsize=6.2, ha="right", va="center",
                    color=(RED if w else "#15803d"), zorder=1.4)
        ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=GREEN, label="정상 AGV", markersize=9),
                           Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, label="고장 경고", markersize=9)],
                  loc="lower left", bbox_to_anchor=(0.004, -0.015), fontsize=7.5, frameon=False, ncol=2)
        return tot

    best = 0.5
    for p in np.linspace(0.05, 0.95, 60):
        if 4 <= sum(sample(a, p)[4] != "정상" for a in plan) <= 7:
            best = p; break
    fig, ax = plt.subplots(figsize=(16, 10.6)); fig.patch.set_facecolor(PAGE)
    render(ax, best); fig.tight_layout()
    fig.savefig(f"{ASSETS}/control_center.png", dpi=125, facecolor=PAGE); plt.close(fig)
    print("저장: control_center.png")

    figg, axg = plt.subplots(figsize=(16, 10.6)); figg.patch.set_facecolor(PAGE)
    NF, ps = 54, np.linspace(0, 1, 54)
    def upd(k):
        axg.clear(); render(axg, ps[k]); return []
    FuncAnimation(figg, upd, frames=NF, interval=110).save(
        f"{ASSETS}/control_center.gif", writer=PillowWriter(fps=10), dpi=66)
    plt.close(figg)
    print(f"저장: control_center.gif ({os.path.getsize(f'{ASSETS}/control_center.gif')/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
