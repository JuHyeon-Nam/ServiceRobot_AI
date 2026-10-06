"""
build_enhanced_dataset.py
-------------------------
원본 라벨링 JSON(zip)에서 '버려졌던' 필드까지 모두 추출해 강화 데이터셋을 만든다.
- 공식 split 사용: Training(TL_) -> train, Validation(VL_) -> val. 자산 중복 검증.
- zip을 디스크에 풀지 않고 메모리 스트리밍으로 파싱
- 동적 센서 7개는 30시점 시퀀스로, 정적/누적/맥락 피처 9개는 윈도우 끝 값으로 부착

출력: --out-dir의 enhanced_{train,val}.npz (X, S, y, groups, target_times)
"""
import argparse
import os, glob, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

WIN = 30
CROWD = {"LOW": 0, "MIDDLE": 1, "HIGH": 2}

# 동적 센서(시점마다 변함) -> 시퀀스로
DYN = ["batteryLevel", "speed", "x", "y", "degree", "collision", "obstacle"]
# 정적/누적/맥락 -> 윈도우 끝 1개 값으로
STAT_NUM = ["isOffline", "nowCharging", "emergencyStop", "batteryUse",
            "batteryCycleCount", "distance", "crowd"]


def f(v, d=0.0):
    try:
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        return float(v)
    except (TypeError, ValueError):
        return d


def parse_record(d):
    dev = d.get("deviceData", {})
    loc = dev.get("standardLocationData", {})
    op = dev.get("totalOperationData", {})
    site = d.get("siteData", {})
    err = d.get("errorData", {})
    rec = {
        "deviceId": d.get("deviceId"),
        "deviceType": d.get("deviceType"),
        "mainState": dev.get("mainState"),
        "createdAt": d.get("createdAt") or d.get("lastUpdateTime") or "",
        "errorCode": err.get("errorCode"),
        # dyn
        "batteryLevel": f(dev.get("batteryLevel")), "speed": f(loc.get("speed")),
        "x": f(loc.get("x")), "y": f(loc.get("y")), "degree": f(loc.get("degree")),
        "collision": f(dev.get("collision")), "obstacle": f(dev.get("obstacle")),
        # stat
        "isOffline": f(dev.get("isOffline")), "nowCharging": f(dev.get("nowCharging")),
        "emergencyStop": f(dev.get("emergencyStop")), "batteryUse": f(op.get("batteryUse")),
        "batteryCycleCount": f(op.get("batteryCycleCount")), "distance": f(op.get("distance")),
        "crowd": float(CROWD.get(site.get("crowd"), 1)),
    }
    return rec


def read_split(folder, tag):
    zips = sorted(glob.glob(os.path.join(folder, "*.zip")))
    if not zips:
        raise FileNotFoundError(f"no AI-Hub JSON archives in {folder}")
    by_dev = defaultdict(list)
    devtypes, mainstates = set(), set()
    n = 0
    for zi, zp in enumerate(zips):
        with zipfile.ZipFile(zp) as zf:
            for nm in zf.namelist():
                if not nm.endswith(".json"):
                    continue
                try:
                    rec = parse_record(json.loads(zf.read(nm).decode("utf-8")))
                except Exception:
                    continue
                if rec["errorCode"] in (None, "NOT_ASSIGNED"):
                    continue
                by_dev[rec["deviceId"]].append(rec)
                devtypes.add(rec["deviceType"]); mainstates.add(rec["mainState"])
                n += 1
        print(f"  [{tag}] {zi+1}/{len(zips)} {os.path.basename(zp)[:40]}  누적 {n}건", flush=True)
    return by_dev, devtypes, mainstates


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build robot windows with asset identity audit.")
    parser.add_argument("--base-dir", required=True, type=Path,
                        help="AI-Hub directory containing Training and Validation")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/datasets/robot-v2")
    args = parser.parse_args(argv)
    BASE, OUT = str(args.base_dir), str(args.out_dir)
    print("== Training 추출 ==", flush=True)
    tr_dev, dt1, ms1 = read_split(os.path.join(BASE, "Training", "02.라벨링데이터"), "train")
    print("== Validation 추출 ==", flush=True)
    va_dev, dt2, ms2 = read_split(os.path.join(BASE, "Validation", "02.라벨링데이터"), "val")
    overlap = set(tr_dev) & set(va_dev)
    if overlap:
        raise ValueError(f"official asset overlap: {sorted(map(str, overlap))[:5]}")

    # 인코더(두 split 공통)
    devtype_map = {v: i for i, v in enumerate(sorted(dt1 | dt2))}
    mainstate_map = {v: i for i, v in enumerate(sorted(s for s in (ms1 | ms2) if s))}
    # 에러코드 인코딩(두 split 공통, 정렬 고정)
    codes = set()
    for dd in (tr_dev, va_dev):
        for recs in dd.values():
            for r in recs:
                codes.add(r["errorCode"])
    err_map = {v: i for i, v in enumerate(sorted(codes))}
    print("errorCode 매핑:", err_map, flush=True)
    print("deviceType:", devtype_map, "| mainState 수:", len(mainstate_map), flush=True)

    def build(by_dev, capture_display=False):
        Xs, Ss, ys, groups, target_times = [], [], [], [], []
        disp = []  # (robot, x, y, degree) at target frame — 대시보드 재생용
        for dev, recs in by_dev.items():
            if not dev:
                raise ValueError("deviceId is required for asset-separated evaluation")
            recs.sort(key=lambda r: r["createdAt"])
            if len(recs) <= WIN:
                continue
            dyn = np.array([[r[c] for c in DYN] for r in recs], dtype=np.float32)
            stat = np.array([[r[c] for c in STAT_NUM] for r in recs], dtype=np.float32)
            dtv = devtype_map.get(recs[0]["deviceType"], 0)
            for i in range(len(recs) - WIN):
                Xs.append(dyn[i:i + WIN])
                end = recs[i + WIN]
                ms = mainstate_map.get(end["mainState"], 0)
                Ss.append(np.concatenate([stat[i + WIN], [dtv, ms]]))
                ys.append(err_map[end["errorCode"]])
                groups.append(str(dev))
                target_times.append(str(end["createdAt"]))
                if capture_display:
                    disp.append((dev, end["deviceType"], end["x"], end["y"],
                                 end["degree"], end["errorCode"]))
        out = (np.asarray(Xs, dtype=np.float32).reshape(-1, WIN, len(DYN)),
               np.asarray(Ss, dtype=np.float32).reshape(-1, 9),
               np.asarray(ys, dtype=np.int64), np.asarray(groups, dtype=str),
               np.asarray(target_times, dtype=str))
        return (out + (disp,)) if capture_display else out

    Xtr, Str, ytr, Gtr, Ttr = build(tr_dev)
    Xva, Sva, yva, Gva, Tva, disp = build(va_dev, capture_display=True)
    if not len(ytr) or not len(yva):
        raise ValueError("both splits must contain sensor windows")
    print(f"train: X{Xtr.shape} S{Str.shape} y{ytr.shape}", flush=True)
    print(f"val:   X{Xva.shape} S{Sva.shape} y{yva.shape}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/enhanced_train.npz", X=Xtr, S=Str, y=ytr, groups=Gtr, target_times=Ttr)
    np.savez_compressed(f"{OUT}/enhanced_val.npz", X=Xva, S=Sva, y=yva, groups=Gva, target_times=Tva)
    json.dump({"err_map": err_map, "devtype_map": devtype_map,
               "mainstate_map": mainstate_map, "crowd_map": CROWD,
               "dyn": DYN, "stat": STAT_NUM + ["deviceType", "mainState"],
               "window_contract": {"history": "30 observations before target", "context": "at target time"},
               "asset_audit": {"train": sorted(map(str, tr_dev)), "validation": sorted(map(str, va_dev)), "overlap": 0}},
              open(f"{OUT}/enhanced_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 대시보드 재생용: val 윈도우와 1:1 정렬된 표시정보(enhanced_val.npz와 index 동일)
    dd = pd.DataFrame(disp, columns=["robot", "deviceType", "x", "y", "degree", "errorCode"])
    dd["validx"] = np.arange(len(dd))
    dd.to_parquet(f"{OUT}/replay_display.parquet", index=False)
    print(f"저장 완료: enhanced_* / replay_display.parquet ({len(dd)}행, {dd.robot.nunique()}대)", flush=True)


if __name__ == "__main__":
    main()
