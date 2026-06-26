"""
make_floorplan.py — 건물 도면 기반 관제 시각화
  - assets/control_center.png : 정적 히어로(도면 + 로봇 + 궤적 트레일)
  - assets/control_center.gif : 로봇이 도면 위를 실제 경로대로 이동하며 AI 진단색이 바뀌는 애니메이션
로봇 위치는 각 구역에 배치된 실제 궤적 모양, 색(정상/경고)은 모델의 실제 예측.
실행: cd src && python make_floorplan.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter

for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False

DATA, ASSETS = "../data/processed", "../assets"
BG, FLOOR, WALL, CORR = "#0a0e1a", "#141c30", "#33415c", "#0f1728"
GREEN, RED, TXT = "#22c55e", "#ef4444", "#cbd5e1"
KOR = {"E-ENV-C": "혼잡·충돌위험", "E-ENV-O": "장애물", "E-INF-A": "자동문연동",
       "E-INF-E": "엘리베이터연동", "E-RBT-B": "배터리저하", "E-RBT-E": "긴급정지",
       "E-RBT-N": "네트워크끊김", "E-RBT-S": "센서이상", "정상": "정상"}

# 도면 구역(x0,y0,w,h,라벨) — 사무동 1층 컨셉
ROOMS = [
    (8, 56, 32, 33, "회의실 A"), (44, 56, 32, 33, "회의실 B"),
    (80, 56, 34, 33, "라운지"), (118, 56, 34, 33, "서버실"),
    (8, 8, 32, 36, "로비 · 입구"), (44, 8, 32, 36, "작업 구역"),
    (80, 8, 34, 36, "물류 창고"), (118, 8, 34, 36, "충전 스테이션"),
]
CORRIDOR = (8, 46, 144, 8)  # 중앙 복도


def draw_floor(ax):
    ax.set_facecolor(BG)
    # 외벽
    ax.add_patch(FancyBboxPatch((5, 5), 150, 86, boxstyle="round,pad=0,rounding_size=2",
                 fc=BG, ec=WALL, lw=2.5, zorder=1))
    # 복도
    cx, cy, cw, ch = CORRIDOR
    ax.add_patch(Rectangle((cx, cy), cw, ch, fc=CORR, ec="none", zorder=1.2))
    ax.text(cx + cw - 2, cy + ch / 2, "중앙 복도", color="#3b4a66", fontsize=7,
            ha="right", va="center", zorder=1.3)
    # 방
    for x0, y0, w, h, label in ROOMS:
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                     fc=FLOOR, ec=WALL, lw=1.4, zorder=1.4))
        ax.text(x0 + 2, y0 + h - 3, label, color="#5b6b8c", fontsize=8.5, va="top", zorder=1.5)
    # 충전 스테이션 표시
    ax.add_patch(Rectangle((120, 10), 6, 4, fc="#f59e0b", ec="none", alpha=0.8, zorder=1.6))
    ax.text(123, 12, "CHG", color="#0a0e1a", fontsize=6, ha="center", va="center", weight="bold", zorder=1.7)
    # 입구 표시(외벽 개구부)
    ax.add_patch(Rectangle((4, 18), 2, 10, fc=BG, ec="none", zorder=1.8))
    ax.text(2, 23, "입구", color="#64748b", fontsize=7, rotation=90, ha="center", va="center", zorder=1.9)
    ax.set_xlim(-2, 158); ax.set_ylim(-2, 96)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    rep = pd.read_parquet(f"{DATA}/replay.parquet")
    # 8개 방 + 복도 2구역 = 최대 10대 배치
    zones = ROOMS + [(10, 46.5, 66, 7, ""), (84, 46.5, 66, 7, "")]
    robots = sorted(rep.robot.unique())[:len(zones)]

    # 로봇별: 실제 궤적을 구역 안으로 정규화 + 예측색
    R = {}
    for rid, (x0, y0, w, h, _) in zip(robots, zones[:len(robots)]):
        g = rep[rep.robot == rid].sort_values("seq").reset_index(drop=True)
        px, py = g.px.to_numpy(), g.py.to_numpy()
        nx = (px - px.min()) / (np.ptp(px) + 1e-9); ny = (py - py.min()) / (np.ptp(py) + 1e-9)
        pad = 0.18
        X = x0 + w * (pad + nx * (1 - 2 * pad)); Y = y0 + h * (pad + ny * (1 - 2 * pad))
        R[rid] = dict(x=X, y=Y, deg=g.degree.to_numpy(), pred=g["pred"].to_numpy(),
                      n=len(g), type=g.deviceType.iloc[0])

    def frame_idx(rid, p):
        return int(p * (R[rid]["n"] - 1))

    def render(ax, p, trails=True):
        draw_floor(ax)
        nwarn = 0
        for rid in robots:
            d = R[rid]; i = frame_idx(rid, p)
            err = d["pred"][i] != "정상"; col = RED if err else GREEN
            nwarn += err
            if trails:  # 최근 궤적 페이드
                a = max(0, i - 12)
                xs, ys = d["x"][a:i + 1], d["y"][a:i + 1]
                for k in range(1, len(xs)):
                    ax.plot(xs[k - 1:k + 1], ys[k - 1:k + 1], color=col,
                            alpha=0.04 + 0.28 * k / len(xs), lw=1.4, zorder=2, solid_capstyle="round")
            x, y = d["x"][i], d["y"][i]
            ax.add_patch(Circle((x, y), 3.6, color=col, alpha=0.16, zorder=2.5))  # glow
            if err:
                ax.add_patch(Circle((x, y), 2.6, fc="none", ec=col, lw=1.6, alpha=0.9, zorder=2.6))
            ax.add_patch(Circle((x, y), 1.5, color=col, zorder=3, ec="white", lw=0.6))
            th = np.deg2rad(d["deg"][i] if not np.isnan(d["deg"][i]) else 0)
            ax.plot([x, x + 3 * np.cos(th)], [y, y + 3 * np.sin(th)], color=col, lw=1.4, zorder=3)
            ax.text(x, y + 4.5, rid, color=TXT, fontsize=6.5, ha="center", zorder=3.2)
            if err:
                ax.text(x, y - 4.8, KOR.get(d["pred"][i], d["pred"][i]),
                        color="#fca5a5", fontsize=6, ha="center", zorder=3.2)
        # 상단 상태바
        ax.add_patch(Rectangle((-2, 90), 160, 8, fc="#0d1424", ec="none", zorder=4))
        ax.text(4, 94, "ROBOT CONTROL CENTER  ·  실시간 예지보전 관제",
                color="#e2e8f0", fontsize=11, weight="bold", va="center", zorder=5)
        ax.text(154, 94, f"가동 {len(robots)}   정상 {len(robots)-nwarn}   경고 {nwarn}",
                color=("#fca5a5" if nwarn else "#86efac"), fontsize=9, ha="right", va="center", zorder=5)
        return nwarn

    # 1) 정적 히어로 — 경고 2~3개 프레임 선택
    best_p = 0.5
    for p in np.linspace(0.1, 0.95, 40):
        if 2 <= sum(R[r]["pred"][frame_idx(r, p)] != "정상" for r in robots) <= 3:
            best_p = p; break
    fig, ax = plt.subplots(figsize=(12, 7.2)); fig.patch.set_facecolor(BG)
    render(ax, best_p)
    fig.tight_layout(); fig.savefig(f"{ASSETS}/control_center.png", dpi=130, facecolor=BG)
    plt.close(fig)
    print("저장: control_center.png")

    # 2) 애니메이션 GIF
    figg, axg = plt.subplots(figsize=(12, 7.2)); figg.patch.set_facecolor(BG)
    NF = 70
    ps = np.linspace(0, 1, NF)

    def upd(k):
        axg.clear(); render(axg, ps[k]); return []
    anim = FuncAnimation(figg, upd, frames=NF, interval=120, blit=False)
    anim.save(f"{ASSETS}/control_center.gif", writer=PillowWriter(fps=9), dpi=80)
    plt.close(figg)
    import os
    print(f"저장: control_center.gif ({os.path.getsize(f'{ASSETS}/control_center.gif')/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
