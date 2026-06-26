"""
build_enhanced_dataset.py
-------------------------
원본 라벨링 JSON(zip)에서 '버려졌던' 필드까지 모두 추출해 강화 데이터셋을 만든다.
- 공식 split 사용: Training(TL_) -> train, Validation(VL_) -> val  (데이터 누수 0)
- zip을 디스크에 풀지 않고 메모리 스트리밍으로 파싱
- 동적 센서 7개는 30시점 시퀀스로, 정적/누적/맥락 피처 9개는 윈도우 끝 값으로 부착

출력: data/processed/enhanced_{train,val}.npz  (X seq, S static, y)
"""
import os, sys, glob, zipfile, json
import numpy as np
from collections import defaultdict

BASE = r"C:\Users\SSAFY\Downloads\42.실내공간 유지관리 서비스 로봇 데이터\3.개방데이터\1.데이터"
OUT = "../data/processed"
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


def main():
    print("== Training 추출 ==", flush=True)
    tr_dev, dt1, ms1 = read_split(os.path.join(BASE, "Training", "02.라벨링데이터"), "train")
    print("== Validation 추출 ==", flush=True)
    va_dev, dt2, ms2 = read_split(os.path.join(BASE, "Validation", "02.라벨링데이터"), "val")

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

    def build(by_dev):
        Xs, Ss, ys = [], [], []
        for dev, recs in by_dev.items():
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
        return (np.asarray(Xs, dtype=np.float32),
                np.asarray(Ss, dtype=np.float32),
                np.asarray(ys, dtype=np.int64))

    Xtr, Str, ytr = build(tr_dev)
    Xva, Sva, yva = build(va_dev)
    print(f"train: X{Xtr.shape} S{Str.shape} y{ytr.shape}", flush=True)
    print(f"val:   X{Xva.shape} S{Sva.shape} y{yva.shape}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/enhanced_train.npz", X=Xtr, S=Str, y=ytr)
    np.savez_compressed(f"{OUT}/enhanced_val.npz", X=Xva, S=Sva, y=yva)
    json.dump({"err_map": err_map, "devtype_map": devtype_map,
               "mainstate_map": mainstate_map, "crowd_map": CROWD,
               "dyn": DYN, "stat": STAT_NUM + ["deviceType", "mainState"]},
              open(f"{OUT}/enhanced_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장 완료: enhanced_train.npz / enhanced_val.npz / enhanced_meta.json", flush=True)


if __name__ == "__main__":
    main()
