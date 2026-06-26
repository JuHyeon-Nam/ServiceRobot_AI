"""
make_visuals.py — README용 시각화 4종 생성 (assets/)
  1) confusion_matrix.png  혼동행렬(행 정규화)
  2) feature_importance.png 피처 중요도 Top12
  3) per_class_f1.png       고장별 F1 + 표본수
  4) dashboard_preview.png  관제 대시보드 프리뷰(플로어 맵 한 프레임)
실행: cd src && python make_visuals.py   (matplotlib 필요)
"""
import json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix, f1_score

# 한글 폰트(Windows 기본 맑은 고딕)
for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False

DATA = "../data/processed"
ASSETS = "../assets"
os.makedirs(ASSETS, exist_ok=True)
MODEL_DYN_IDX = [0, 1, 4, 5, 6]
KOR = {"E-ENV-C": "혼잡·충돌위험", "E-ENV-O": "장애물", "E-INF-A": "자동문연동",
       "E-INF-E": "엘리베이터연동", "E-RBT-B": "배터리저하", "E-RBT-E": "긴급정지",
       "E-RBT-N": "네트워크끊김", "E-RBT-S": "센서이상", "정상": "정상"}

booster = lgb.Booster(model_file=f"{DATA}/robot_pdm_enhanced.txt")
mm = json.load(open(f"{DATA}/robot_pdm_enhanced_meta.json", encoding="utf-8"))
em = json.load(open(f"{DATA}/enhanced_meta.json", encoding="utf-8"))
names, classes = mm["class_names"], np.array(mm["classes"])
labels = [f"{c}\n{KOR.get(c,'')}" for c in names]


def feat(X, S):
    X = X[:, :, MODEL_DYN_IDX]; N = X.shape[0]
    return np.hstack([X.reshape(N, -1), X.mean(1), X.std(1),
        X[:, -10:, :].mean(1) - X[:, :10, :].mean(1),
        np.abs(np.fft.rfft(X, axis=1))[:, :15, :].reshape(N, -1), S]).astype(np.float32)


va = np.load(f"{DATA}/enhanced_val.npz")
proba = booster.predict(feat(va["X"], va["S"]))
yp = classes[proba.argmax(1)]; y = va["y"]

# 1) 혼동행렬 -------------------------------------------------
cm = confusion_matrix(y, yp, labels=list(range(len(names))))
cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
ax.set_yticklabels(labels, fontsize=8)
for i in range(len(names)):
    for j in range(len(names)):
        if cm[i, j]:
            ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if cmn[i, j] > 0.5 else "#333")
ax.set_xlabel("예측"); ax.set_ylabel("실제")
ax.set_title("혼동행렬 (행 정규화 %) · 공식 Validation", fontsize=12, weight="bold")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(f"{ASSETS}/confusion_matrix.png", dpi=130); plt.close(fig)

# 2) 피처 중요도 Top12 ---------------------------------------
mdyn = [em["dyn"][i] for i in MODEL_DYN_IDX]
eng_names = ([f"{s}_t{t}" for t in range(30) for s in mdyn]
             + [f"{s}_mean" for s in mdyn] + [f"{s}_std" for s in mdyn]
             + [f"{s}_trend" for s in mdyn]
             + [f"fft{k}_{s}" for k in range(15) for s in mdyn] + em["stat"])
imp = booster.feature_importance()
order = np.argsort(imp)[::-1][:12]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([eng_names[i] for i in order][::-1], [imp[i] for i in order][::-1], color="#2563eb")
ax.set_title("피처 중요도 Top12 (1·2·3위 = 복원한 누적·상태 신호)", fontsize=12, weight="bold")
ax.set_xlabel("중요도 (split gain)")
fig.tight_layout(); fig.savefig(f"{ASSETS}/feature_importance.png", dpi=130); plt.close(fig)

# 3) 클래스별 F1 --------------------------------------------
f1s = f1_score(y, yp, average=None, labels=list(range(len(names))))
sup = [int((y == i).sum()) for i in range(len(names))]
col = ["#16a34a" if f >= 0.85 else "#f59e0b" if f >= 0.5 else "#dc2626" for f in f1s]
fig, ax = plt.subplots(figsize=(9, 5))
b = ax.bar(range(len(names)), f1s, color=col)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([f"{KOR.get(c,c)}\n(n={s})" for c, s in zip(names, sup)], fontsize=8, rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score")
ax.axhline(0.85, ls="--", color="#16a34a", lw=0.8)
ax.set_title("고장별 진단 성능(F1) · 초록=우수, 빨강=표본부족(센서한계)", fontsize=12, weight="bold")
for i, f in enumerate(f1s):
    ax.text(i, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(f"{ASSETS}/per_class_f1.png", dpi=130); plt.close(fig)

# 4) 대시보드 프리뷰(플로어 맵 한 프레임) ---------------------
rep = pd.read_parquet(f"{DATA}/replay.parquet")
# 경고가 2~4개인 흥미로운 프레임 선택
cnt = rep[rep.pred != "정상"].groupby("seq").size()
good = cnt[(cnt >= 2) & (cnt <= 4)]
seq = int(good.index[len(good)//2]) if len(good) else (int(cnt.idxmax()) if len(cnt) else int(rep.seq.median()))
cur = rep[rep.seq == seq].copy().reset_index(drop=True)
# 정적 히어로: 명확함 우선 — 로봇을 5x2 격자에 배치(실제 진행각·상태 유지)
robots_sorted = sorted(cur.robot.unique())
ncol = 5
anchors = {r: (12 + (i % ncol) * 19, 70 - (i // ncol) * 32) for i, r in enumerate(robots_sorted)}
fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor("#0b1220"); ax.set_facecolor("#0b1220")
for g in range(0, 101, 20):
    ax.axvline(g, color="#162033", lw=0.8); ax.axhline(g, color="#162033", lw=0.8)
for _, r in cur.iterrows():
    ax_, ay_ = anchors[r.robot]
    err = r["pred"] != "정상"; c = "#ef4444" if err else "#22c55e"
    th = np.deg2rad(r["degree"] if pd.notna(r["degree"]) else 0)
    ax.quiver(ax_, ay_, np.cos(th), np.sin(th), color=c, scale=26, width=0.008,
              headwidth=4.5, headlength=5.5, zorder=3)
    ax.add_patch(plt.Circle((ax_, ay_), 7.5, color=c, alpha=0.12, zorder=1))
    ax.text(ax_, ay_ + 9, r.robot, color="#e2e8f0", fontsize=8, ha="center", zorder=4)
    tag = f"{KOR.get(r['pred'],'정상')}" if not err else f"[경고] {KOR.get(r['pred'],r['pred'])}"
    ax.text(ax_, ay_ - 10, tag, color=("#fca5a5" if err else "#86efac"), fontsize=7, ha="center", zorder=4)
ax.set_xlim(0, 100); ax.set_ylim(8, 92); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("서비스 로봇 실시간 관제 — 디지털 트윈  (초록=정상 / 빨강=고장경고)",
             color="#e2e8f0", fontsize=12, weight="bold")
for s in ax.spines.values():
    s.set_color("#334155")
fig.tight_layout(); fig.savefig(f"{ASSETS}/dashboard_preview.png", dpi=130, facecolor="#0b1220"); plt.close(fig)

print("생성 완료:", os.listdir(ASSETS))
