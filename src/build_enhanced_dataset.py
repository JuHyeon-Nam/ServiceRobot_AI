"""
build_enhanced_dataset.py
-------------------------
원본 라벨링 JSON(zip)에서 '버려졌던' 필드까지 모두 추출해 강화 데이터셋을 만든다.
- 공식 split 사용: Training(TL_) -> train, Validation(VL_) -> val. 자산 중복 검증.
- zip을 디스크에 풀지 않고 메모리 스트리밍으로 파싱
- 동적 센서 7개는 30시점 시퀀스로, 정적/누적/맥락 피처 9개는 윈도우 끝 값으로 부착

출력: 새 --out-dir의 enhanced_{train,val}.npz (X, S, y, groups, target_times)
"""
import argparse
from math import isfinite
from numbers import Real
import os, glob, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

WIN = 30
CROWD = {"LOW": 0, "MIDDLE": 1, "HIGH": 2}
INGEST_SKIP_REASONS = ("invalid_utf8", "invalid_json", "record_parse_error",
                       "missing_error_code", "not_assigned_error_code")

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


def read_split(folder, tag, *, audit=None):
    """Read JSON ZIP entries once; publish counts only after a complete split."""
    zips = sorted(glob.glob(os.path.join(folder, "*.zip")))
    if not zips:
        raise FileNotFoundError(f"no AI-Hub JSON archives in {folder}")
    by_dev = defaultdict(list)
    devtypes, mainstates = set(), set()
    n, archives = 0, []
    for zi, zp in enumerate(zips):
        entry = {"archive": os.path.basename(zp), "members": 0, "json_members": 0,
                 "non_json_members": 0, "accepted_records": 0, "accepted_by_class": {},
                 "skipped_json": dict.fromkeys(INGEST_SKIP_REASONS, 0)}
        with zipfile.ZipFile(zp) as zf:
            for member in zf.infolist():
                entry["members"] += 1
                if not member.filename.endswith(".json"):
                    entry["non_json_members"] += 1
                    continue
                entry["json_members"] += 1
                # Archive/CRC/read failures are not malformed-record skips.
                payload = zf.read(member)
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    entry["skipped_json"]["invalid_utf8"] += 1
                    continue
                try:
                    document = json.loads(text)
                except json.JSONDecodeError:
                    entry["skipped_json"]["invalid_json"] += 1
                    continue
                try:
                    rec = parse_record(document)
                except (AttributeError, TypeError):
                    entry["skipped_json"]["record_parse_error"] += 1
                    continue
                if rec["errorCode"] is None:
                    entry["skipped_json"]["missing_error_code"] += 1
                    continue
                if rec["errorCode"] == "NOT_ASSIGNED":
                    entry["skipped_json"]["not_assigned_error_code"] += 1
                    continue
                by_dev[rec["deviceId"]].append(rec)
                devtypes.add(rec["deviceType"]); mainstates.add(rec["mainState"])
                code = rec["errorCode"]
                entry["accepted_by_class"][code] = entry["accepted_by_class"].get(code, 0) + 1
                entry["accepted_records"] += 1
                n += 1
        archives.append(entry)
        skipped = sum(entry["skipped_json"].values())
        print(f"  [{tag}] {zi+1}/{len(zips)} {entry['archive'][:40]}  "
              f"JSON {entry['json_members']}건 / 제외 {skipped}건 / 누적 채택 {n}건", flush=True)
    summary = {key: sum(entry[key] for entry in archives)
               for key in ("members", "json_members", "non_json_members", "accepted_records")}
    class_counts = Counter()
    for entry in archives:
        class_counts.update(entry["accepted_by_class"])
    summary.update({"archive_count": len(archives), "archives": archives,
                    "accepted_by_class": dict(class_counts),
                    "skipped_json": {reason: sum(entry["skipped_json"][reason] for entry in archives)
                                     for reason in INGEST_SKIP_REASONS}})
    print(f"  [{tag}] JSON 제외 사유: {summary['skipped_json']}", flush=True)
    if audit is not None:
        audit.update(summary)
    return by_dev, devtypes, mainstates


def _gap_threshold(seconds):
    if seconds is None:
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, Real) or not isfinite(seconds) or seconds <= 0:
        raise ValueError("max gap seconds must be a finite positive number")
    try:
        threshold = pd.Timedelta(seconds=float(seconds))
    except (ValueError, OverflowError) as error:
        raise ValueError("max gap seconds must be representable as a pandas duration") from error
    if threshold <= pd.Timedelta(0):
        raise ValueError("max gap seconds must be at least one nanosecond")
    return threshold


def prepare_sequence(records, device_id, max_gap_seconds=None):
    """Order an unambiguous asset clock and mark explicitly configured session gaps."""
    if (not isinstance(device_id, str) or not device_id.strip() or device_id != device_id.strip()
            or device_id.lower() in {"none", "null", "nan", "nat"}):
        raise ValueError("deviceId must be a nonempty, trimmed, nonmissing string")
    threshold = _gap_threshold(max_gap_seconds)
    if not records:
        raise ValueError(f"no timestamped records for {device_id}")
    times = []
    for record in records:
        value = record.get("createdAt")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing createdAt for {device_id}")
        try:
            timestamp = pd.to_datetime(value.strip(), format="ISO8601", errors="raise")
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"invalid ISO8601 createdAt for {device_id}") from error
        if pd.isna(timestamp):
            raise ValueError(f"invalid ISO8601 createdAt for {device_id}")
        times.append(timestamp)
    aware = [timestamp.tzinfo is not None for timestamp in times]
    if any(aware) and not all(aware):
        raise ValueError(f"mixed timezone-aware and naive createdAt for {device_id}")
    basis = "utc" if all(aware) else "naive_shared_clock"
    if all(aware):
        times = [timestamp.tz_convert("UTC") for timestamp in times]
    ordered = sorted(zip(times, records), key=lambda pair: pair[0])
    times, records = [pair[0] for pair in ordered], [pair[1] for pair in ordered]
    gaps = np.zeros(len(records), dtype=np.int64)
    for i in range(1, len(times)):
        if times[i] == times[i - 1]:
            raise ValueError(f"duplicate createdAt for {device_id}: {times[i].isoformat()}")
        if threshold is not None and times[i] - times[i - 1] > threshold:
            gaps[i] = 1
    return records, np.cumsum(gaps), {"timestamp_basis": basis, "gaps": int(gaps.sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build robot windows with asset identity audit.")
    parser.add_argument("--base-dir", required=True, type=Path,
                        help="AI-Hub directory containing Training and Validation")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/datasets/robot-v2",
                        help="new directory for this build; existing paths and symlinks are rejected")
    parser.add_argument("--max-gap-seconds", type=float,
                        help="exclude windows crossing larger observation gaps; unset disables gap filtering")
    args = parser.parse_args(argv)
    try:
        _gap_threshold(args.max_gap_seconds)
    except ValueError as error:
        parser.error(str(error))
    BASE, OUT = str(args.base_dir), args.out_dir
    if OUT.exists() or OUT.is_symlink():
        parser.error(f"--out-dir must be a new directory, not an existing path or symlink: {OUT}")
    train_ingestion, validation_ingestion = {}, {}
    print("== Training 추출 ==", flush=True)
    tr_dev, dt1, ms1 = read_split(os.path.join(BASE, "Training", "02.라벨링데이터"), "train", audit=train_ingestion)
    print("== Validation 추출 ==", flush=True)
    va_dev, dt2, ms2 = read_split(os.path.join(BASE, "Validation", "02.라벨링데이터"), "val", audit=validation_ingestion)
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

    sequence_audit = {"max_gap_seconds": args.max_gap_seconds,
                      "timestamp_policy": "consistent_asset_clock; aware_times_normalized_to_utc",
                      "duplicate_policy": "reject", "splits": {}}

    def build(by_dev, tag, capture_display=False):
        Xs, Ss, ys, groups, target_times = [], [], [], [], []
        disp = []  # (robot, x, y, degree) at target frame — 대시보드 재생용
        audit = {"records": 0, "input_assets": len(by_dev), "gaps": 0, "candidate_windows": 0,
                 "discarded_gap_windows": 0, "emitted_windows": 0,
                 "timestamp_basis_assets": {"utc": 0, "naive_shared_clock": 0}}
        sequence_audit["splits"][tag] = audit
        for dev, recs in by_dev.items():
            recs, sessions, clock = prepare_sequence(recs, dev, args.max_gap_seconds)
            audit["records"] += len(recs)
            audit["gaps"] += clock["gaps"]
            audit["timestamp_basis_assets"][clock["timestamp_basis"]] += 1
            audit["candidate_windows"] += max(0, len(recs) - WIN)
            if len(recs) <= WIN:
                continue
            dyn = np.array([[r[c] for c in DYN] for r in recs], dtype=np.float32)
            stat = np.array([[r[c] for c in STAT_NUM] for r in recs], dtype=np.float32)
            dtv = devtype_map.get(recs[0]["deviceType"], 0)
            for i in range(len(recs) - WIN):
                # Include the target observation in continuity checks, not only its 30-row history.
                if sessions[i] != sessions[i + WIN]:
                    audit["discarded_gap_windows"] += 1
                    continue
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
        audit["emitted_windows"] = len(ys)
        out = (np.asarray(Xs, dtype=np.float32).reshape(-1, WIN, len(DYN)),
               np.asarray(Ss, dtype=np.float32).reshape(-1, 9),
               np.asarray(ys, dtype=np.int64), np.asarray(groups, dtype=str),
               np.asarray(target_times, dtype=str))
        return (out + (disp,)) if capture_display else out

    Xtr, Str, ytr, Gtr, Ttr = build(tr_dev, "train")
    Xva, Sva, yva, Gva, Tva, disp = build(va_dev, "validation", capture_display=True)
    if not len(ytr) or not len(yva):
        raise ValueError("both splits must contain sensor windows")
    print(f"train: X{Xtr.shape} S{Str.shape} y{ytr.shape}", flush=True)
    print(f"val:   X{Xva.shape} S{Sva.shape} y{yva.shape}", flush=True)

    # Claim the destination again after parsing; never merge with another build's files.
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "enhanced_train.npz").open("xb") as stream:
        np.savez_compressed(stream, X=Xtr, S=Str, y=ytr, groups=Gtr, target_times=Ttr)
    with (OUT / "enhanced_val.npz").open("xb") as stream:
        np.savez_compressed(stream, X=Xva, S=Sva, y=yva, groups=Gva, target_times=Tva)

    # 대시보드 재생용: val 윈도우와 1:1 정렬된 표시정보(enhanced_val.npz와 index 동일)
    dd = pd.DataFrame(disp, columns=["robot", "deviceType", "x", "y", "degree", "errorCode"])
    dd["validx"] = np.arange(len(dd))
    with (OUT / "replay_display.parquet").open("xb") as stream:
        dd.to_parquet(stream, index=False)
    # Publish metadata last so earlier write failures cannot leave usable build metadata.
    with (OUT / "enhanced_meta.json").open("x", encoding="utf-8") as stream:
        json.dump({"err_map": err_map, "devtype_map": devtype_map,
                   "mainstate_map": mainstate_map, "crowd_map": CROWD,
                   "dyn": DYN, "stat": STAT_NUM + ["deviceType", "mainState"],
                   "window_contract": {"history": "30 observations before target", "context": "at target time"},
                   "ingestion_audit": {"schema": "servicerobot.dataset.ingestion.v1",
                                       "json_member_policy": "case_sensitive_.json_suffix; each_zip_entry",
                                       "splits": {"train": train_ingestion, "validation": validation_ingestion}},
                   "sequence_audit": sequence_audit,
                   "asset_audit": {"train": sorted(map(str, tr_dev)), "validation": sorted(map(str, va_dev)), "overlap": 0}},
                  stream, ensure_ascii=False, indent=1)
    print(f"저장 완료: enhanced_* / replay_display.parquet ({len(dd)}행, {dd.robot.nunique()}대)", flush=True)


if __name__ == "__main__":
    main()
