"""
build_replay.py
---------------
관제 대시보드용 재생 데이터를 만든다.
- enhanced_val.npz(모델이 보는 바로 그 윈도우) + replay_display.parquet(1:1 표시정보)을 결합.
- 모델 예측을 '미리' 계산해 저장 → 대시보드는 추론 없이 가볍게 재생.
- 진단 정확도가 양호하고 에러가 섞인 로봇들을 골라 현실적인 관제 화면을 구성.
선행: build_enhanced_dataset.py (enhanced_val.npz, replay_display.parquet 생성)
출력: data/processed/replay.parquet, replay_meta.json
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

DATA = "../data/processed"
MODEL_DYN_IDX = [0, 1, 4, 5, 6]      # x,y 제외(학습과 동일)
N_ROBOTS = 10
MAX_FRAMES = 600                      # 로봇당 재생 프레임 상한(너무 길면 다운샘플)


def feat(X, S):
    X = X[:, :, MODEL_DYN_IDX]; N = X.shape[0]
    return np.hstack([X.reshape(N, -1), X.mean(1), X.std(1),
        X[:, -10:, :].mean(1) - X[:, :10, :].mean(1),
        np.abs(np.fft.rfft(X, axis=1))[:, :15, :].reshape(N, -1), S]).astype(np.float32)


def main():
    booster = lgb.Booster(model_file=f"{DATA}/robot_pdm_enhanced.txt")
    mm = json.load(open(f"{DATA}/robot_pdm_enhanced_meta.json", encoding="utf-8"))
    names, classes = mm["class_names"], np.array(mm["classes"])
    va = np.load(f"{DATA}/enhanced_val.npz")
    dd = pd.read_parquet(f"{DATA}/replay_display.parquet")

    proba = booster.predict(feat(va["X"], va["S"]))
    top = proba.argmax(1)
    dd["pred"] = [names[int(classes[t])] for t in top]
    dd["conf"] = proba[np.arange(len(top)), top]
    dd["ok"] = dd["pred"] == dd["errorCode"]

    # 로봇 선별: 정확도 양호 + 에러 보유(흥미로운 경고) 우선, 타입 다양성
    g = dd.groupby("robot").agg(n=("validx", "size"), acc=("ok", "mean"),
        type=("deviceType", "first"), nerr=("errorCode", lambda s: (s != "정상").sum()))
    g = g[(g.n >= 50) & (g.acc >= 0.80)].reset_index()
    # 점수: 에러 있는 로봇 우대 + 정확도
    g["score"] = g.acc + (g.nerr > 0) * 0.5
    g = g.sort_values("score", ascending=False)
    picked, seen = [], {}
    for _, r in g.iterrows():
        if seen.get(r.type, 0) >= 3:
            continue
        picked.append(r.robot); seen[r.type] = seen.get(r.type, 0) + 1
        if len(picked) >= N_ROBOTS:
            break
    print("선별 로봇:", picked)

    sub = dd[dd.robot.isin(picked)].copy().sort_values(["robot", "validx"])
    # 로봇별 seq(0..) 부여 + 길면 균등 다운샘플
    out = []
    for rid, grp in sub.groupby("robot"):
        grp = grp.reset_index(drop=True)
        if len(grp) > MAX_FRAMES:
            grp = grp.iloc[np.linspace(0, len(grp) - 1, MAX_FRAMES).astype(int)].reset_index(drop=True)
        grp["seq"] = np.arange(len(grp))
        out.append(grp)
    rep = pd.concat(out, ignore_index=True)

    # 좌표 정규화(여러 로봇을 한 가상 플로어로)
    xmin, xmax, ymin, ymax = rep.x.min(), rep.x.max(), rep.y.min(), rep.y.max()
    rep["px"] = (rep.x - xmin) / (xmax - xmin + 1e-9) * 100
    rep["py"] = (rep.y - ymin) / (ymax - ymin + 1e-9) * 100
    rep = rep[["robot", "deviceType", "seq", "px", "py", "degree", "pred", "conf", "errorCode"]]
    rep.to_parquet(f"{DATA}/replay.parquet", index=False)
    json.dump({"robots": picked, "n_frames_max": int(rep.seq.max())},
              open(f"{DATA}/replay_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    acc = (rep.pred == rep.errorCode).mean()
    print(f"저장 replay.parquet ({len(rep)}행, {rep.robot.nunique()}대, 진단정확도 {acc*100:.1f}%, "
          f"경고프레임 {(rep.pred!='정상').sum()})")


if __name__ == "__main__":
    main()
