"""
build_enhanced_dataset.py
-------------------------
원본 라벨링 JSON(zip)에서 '버려졌던' 필드까지 모두 추출해 강화 데이터셋을 만든다.
- 공식 split 사용: Training(TL_) -> train, Validation(VL_) -> val. 자산 중복 검증.
- zip을 디스크에 풀지 않고 메모리 스트리밍으로 파싱
- 동적 센서 7개는 30시점 시퀀스로, 정적/누적/맥락 피처 9개는 윈도우 끝 값으로 부착

출력: 새 --out-dir의 enhanced_{train,val}.npz (X, S, y, groups, target_times)
--training-only: 고정 인코딩으로 Training만 생성. 공식 Validation/재생 파일은 읽거나 만들지 않음.
"""
import argparse
import hashlib
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


def read_split(folder, tag, *, audit=None, reject_malformed_records=False,
               include_unassigned_history=False, encoding_metadata=None):
    """Read JSON ZIP entries once; publish counts only after a complete split."""
    if include_unassigned_history:
        if tag != "train" or encoding_metadata is None or not reject_malformed_records:
            raise ValueError("unassigned history requires strict Training input and frozen encoding metadata")
        from diagnosis_record_quality import record_quality
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
        if include_unassigned_history:
            entry["history_only_records"] = 0
        with zipfile.ZipFile(zp) as zf:
            for member in zf.infolist():
                entry["members"] += 1
                if not member.filename.endswith(".json"):
                    entry["non_json_members"] += 1
                    continue
                entry["json_members"] += 1
                if include_unassigned_history and member.file_size > 8 * 1024 * 1024:
                    raise ValueError(f"JSON member exceeds experimental history budget: {zp}::{member.filename}")
                # Archive/CRC/read failures are not malformed-record skips.
                payload = zf.read(member)
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError as error:
                    entry["skipped_json"]["invalid_utf8"] += 1
                    if reject_malformed_records:
                        raise ValueError(f"invalid_utf8 in {zp}::{member.filename}; malformed records rejected") from error
                    continue
                try:
                    document = json.loads(text)
                except json.JSONDecodeError as error:
                    entry["skipped_json"]["invalid_json"] += 1
                    if reject_malformed_records:
                        raise ValueError(f"invalid_json in {zp}::{member.filename}; malformed records rejected") from error
                    continue
                try:
                    rec = parse_record(document)
                except (AttributeError, TypeError) as error:
                    entry["skipped_json"]["record_parse_error"] += 1
                    if reject_malformed_records:
                        raise ValueError(f"record_parse_error in {zp}::{member.filename}; malformed records rejected") from error
                    continue
                if rec["errorCode"] is None:
                    entry["skipped_json"]["missing_error_code"] += 1
                    continue
                unassigned = rec["errorCode"] == "NOT_ASSIGNED"
                if unassigned and not include_unassigned_history:
                    entry["skipped_json"]["not_assigned_error_code"] += 1
                    continue
                if include_unassigned_history:
                    quality = record_quality(document, rec, encoding_metadata)
                    if not quality["dynamic_usable"] or (not unassigned and not quality["target_context_usable"]):
                        raise ValueError(f"raw sensor/context quality rejected in {zp}::{member.filename}: {quality}")
                    entry["history_only_records"] += int(unassigned)
                by_dev[rec["deviceId"]].append(rec)
                if not (include_unassigned_history and unassigned):
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
                    "malformed_record_policy": "reject" if reject_malformed_records else "count_and_skip",
                    "accepted_by_class": dict(class_counts),
                    "skipped_json": {reason: sum(entry["skipped_json"][reason] for entry in archives)
                                     for reason in INGEST_SKIP_REASONS}})
    if include_unassigned_history:
        summary["history_only_records"] = sum(entry["history_only_records"] for entry in archives)
        summary["target_labeled_records"] = n - summary["history_only_records"]
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


def load_encoding_metadata(path: Path):
    """Freeze an explicit encoding contract; hash the exact bytes used for parsing."""
    raw = path.read_bytes()
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("encoding metadata must be a JSON object")
    if metadata.get("dyn") != DYN or metadata.get("stat") != STAT_NUM + ["deviceType", "mainState"]:
        raise ValueError("encoding metadata sensor/context order differs from the builder contract")
    crowd = metadata.get("crowd_map")
    if not isinstance(crowd, dict) or crowd != CROWD or any(type(value) is not int for value in crowd.values()):
        raise ValueError("encoding metadata crowd map differs from the parser contract")
    for name in ("err_map", "devtype_map", "mainstate_map"):
        mapping = metadata.get(name)
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(key, str) or not key.strip() or key != key.strip() for key in mapping)
                or any(type(value) is not int for value in mapping.values())
                or sorted(mapping.values()) != list(range(len(mapping)))):
            raise ValueError(f"encoding metadata {name} must use nonempty categories and contiguous integer indices")
    if "정상" not in metadata["err_map"]:
        raise ValueError("encoding metadata err_map must include normal")
    return metadata, {"mode": "frozen_metadata", "source_path": str(path.resolve()),
                      "source_sha256": hashlib.sha256(raw).hexdigest(),
                      "unknown_category_policy": "reject_deviceType_mainState_errorCode"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build robot windows with asset identity audit.")
    parser.add_argument("--base-dir", required=True, type=Path,
                        help="AI-Hub directory containing Training and Validation")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/datasets/robot-v2",
                        help="new directory for this build; existing paths and symlinks are rejected")
    parser.add_argument("--max-gap-seconds", type=float,
                        help="exclude windows crossing larger observation gaps; unset disables gap filtering")
    parser.add_argument("--reject-malformed-records", action="store_true",
                        help="abort on UTF-8, JSON syntax or known record parsing failures; "
                             "label filters and numeric defaults are unchanged")
    parser.add_argument("--encoding-meta", type=Path,
                        help="reuse fixed maps from this metadata; reject unknown deviceType/mainState/errorCode")
    parser.add_argument("--training-only", action="store_true",
                        help="build Training windows only for internal robot holdout; requires --encoding-meta; "
                             "do not read official Validation or create validation/replay files")
    parser.add_argument("--include-unassigned-history", action="store_true",
                        help="experimental Training-only mode: NOT_ASSIGNED sensors may enter history, never targets; "
                             "requires frozen encoding, explicit gap limit and strict malformed-record rejection")
    args = parser.parse_args(argv)
    if args.training_only and args.encoding_meta is None:
        parser.error("--training-only requires --encoding-meta to preserve the existing encoding contract")
    if args.include_unassigned_history and (not args.training_only or args.max_gap_seconds is None
                                            or not args.reject_malformed_records):
        parser.error("--include-unassigned-history requires --training-only, --encoding-meta, "
                     "--max-gap-seconds and --reject-malformed-records")
    try:
        _gap_threshold(args.max_gap_seconds)
    except ValueError as error:
        parser.error(str(error))
    BASE, OUT = str(args.base_dir), args.out_dir
    if OUT.exists() or OUT.is_symlink():
        parser.error(f"--out-dir must be a new directory, not an existing path or symlink: {OUT}")
    encodings = None
    encoding_audit = {"mode": "fit_on_official_split_union", "source_path": None, "source_sha256": None,
                      "unknown_category_policy": "legacy_defaults"}
    if args.encoding_meta is not None:
        encodings, encoding_audit = load_encoding_metadata(args.encoding_meta)
    history_options, history_sources = {}, None
    if args.include_unassigned_history:
        from audit_diagnosis_continuity import _fingerprints
        history_sources = _fingerprints(Path(BASE) / "Training/02.라벨링데이터")
        history_options = {"include_unassigned_history": True, "encoding_metadata": encodings}
    train_ingestion, validation_ingestion = {}, {}
    print("== Training 추출 ==", flush=True)
    tr_dev, dt1, ms1 = read_split(os.path.join(BASE, "Training", "02.라벨링데이터"), "train", audit=train_ingestion,
                                 reject_malformed_records=args.reject_malformed_records, **history_options)
    va_dev, dt2, ms2 = {}, set(), set()
    if args.training_only:
        print("== Training 전용: 공식 Validation 읽기/중복 검사/재생 생성 미수행 ==", flush=True)
    else:
        print("== Validation 추출 ==", flush=True)
        va_dev, dt2, ms2 = read_split(os.path.join(BASE, "Validation", "02.라벨링데이터"), "val", audit=validation_ingestion,
                                     reject_malformed_records=args.reject_malformed_records)
        overlap = set(tr_dev) & set(va_dev)
        if overlap:
            raise ValueError(f"official asset overlap: {sorted(map(str, overlap))[:5]}")

    if encodings is None:
        # 인코더(두 split 공통, 기존 동작)
        devtype_map = {v: i for i, v in enumerate(sorted(dt1 | dt2))}
        mainstate_map = {v: i for i, v in enumerate(sorted(s for s in (ms1 | ms2) if s))}
        codes = set()
        for dd in (tr_dev, va_dev):
            for recs in dd.values():
                for r in recs:
                    codes.add(r["errorCode"])
        err_map = {v: i for i, v in enumerate(sorted(codes))}
    else:
        devtype_map, mainstate_map, err_map = [encodings[key] for key in ("devtype_map", "mainstate_map", "err_map")]
        # Check every accepted observation, including short histories that emit no windows.
        for tag, by_dev in (("train", tr_dev), ("validation", va_dev)):
            for device, records in by_dev.items():
                for record in records:
                    if args.include_unassigned_history and record["errorCode"] == "NOT_ASSIGNED":
                        continue
                    for field, mapping in (("deviceType", devtype_map), ("mainState", mainstate_map), ("errorCode", err_map)):
                        value = record[field]
                        if not isinstance(value, str) or value not in mapping:
                            raise ValueError(f"unknown {field} {value!r} in {tag} for {device}; frozen encoding")
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
        if args.include_unassigned_history:
            audit.update({"unassigned_target_windows": 0, "labeled_candidate_windows": 0,
                          "emitted_windows_with_unassigned_history": 0})
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
            stat = None if args.include_unassigned_history else np.array([[r[c] for c in STAT_NUM] for r in recs], dtype=np.float32)
            dtv = None if args.include_unassigned_history else devtype_map.get(recs[0]["deviceType"], 0)
            if args.include_unassigned_history:
                history_prefix = np.concatenate(([0], np.cumsum([r["errorCode"] == "NOT_ASSIGNED" for r in recs])))
            for i in range(len(recs) - WIN):
                end = recs[i + WIN]
                if args.include_unassigned_history:
                    if end["errorCode"] == "NOT_ASSIGNED":
                        audit["unassigned_target_windows"] += 1
                        continue
                    audit["labeled_candidate_windows"] += 1
                # Include the target observation in continuity checks, not only its 30-row history.
                if sessions[i] != sessions[i + WIN]:
                    audit["discarded_gap_windows"] += 1
                    continue
                Xs.append(dyn[i:i + WIN])
                ms = mainstate_map.get(end["mainState"], 0)
                target_stat = np.array([end[c] for c in STAT_NUM], dtype=np.float32) if args.include_unassigned_history else stat[i + WIN]
                target_type = devtype_map[end["deviceType"]] if args.include_unassigned_history else dtv
                Ss.append(np.concatenate([target_stat, [target_type, ms]]))
                if args.include_unassigned_history:
                    audit["emitted_windows_with_unassigned_history"] += int(history_prefix[i + WIN] > history_prefix[i])
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
    if args.training_only:
        if not len(ytr):
            raise ValueError("training split must contain sensor windows")
    else:
        Xva, Sva, yva, Gva, Tva, disp = build(va_dev, "validation", capture_display=True)
        if not len(ytr) or not len(yva):
            raise ValueError("both splits must contain sensor windows")
    print(f"train: X{Xtr.shape} S{Str.shape} y{ytr.shape}", flush=True)
    if not args.training_only:
        print(f"val:   X{Xva.shape} S{Sva.shape} y{yva.shape}", flush=True)

    if args.include_unassigned_history and history_sources != _fingerprints(Path(BASE) / "Training/02.라벨링데이터"):
        raise ValueError("Training archives changed during experimental history build")
    # Claim the destination again after parsing; never merge with another build's files.
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "enhanced_train.npz").open("xb") as stream:
        np.savez_compressed(stream, X=Xtr, S=Str, y=ytr, groups=Gtr, target_times=Ttr)
    if not args.training_only:
        with (OUT / "enhanced_val.npz").open("xb") as stream:
            np.savez_compressed(stream, X=Xva, S=Sva, y=yva, groups=Gva, target_times=Tva)

    # 대시보드 재생용: val 윈도우와 1:1 정렬된 표시정보(enhanced_val.npz와 index 동일)
    if not args.training_only:
        dd = pd.DataFrame(disp, columns=["robot", "deviceType", "x", "y", "degree", "errorCode"])
        dd["validx"] = np.arange(len(dd))
        with (OUT / "replay_display.parquet").open("xb") as stream:
            dd.to_parquet(stream, index=False)
    ingestion_splits = {"train": train_ingestion}
    asset_audit = {"train": sorted(map(str, tr_dev)), "validation": None, "overlap": None}
    if not args.training_only:
        ingestion_splits["validation"] = validation_ingestion
        asset_audit.update({"validation": sorted(map(str, va_dev)), "overlap": 0})
    metadata = {"err_map": err_map, "devtype_map": devtype_map,
                "mainstate_map": mainstate_map, "crowd_map": CROWD,
                "dyn": DYN, "stat": STAT_NUM + ["deviceType", "mainState"],
                "window_contract": {"history": "30 observations before target", "context": "at target time"},
                "encoding_audit": encoding_audit,
                "ingestion_audit": {"schema": "servicerobot.dataset.ingestion.v1",
                                    "json_member_policy": "case_sensitive_.json_suffix; each_zip_entry",
                                    "splits": ingestion_splits},
                "sequence_audit": sequence_audit, "asset_audit": asset_audit}
    if args.training_only:
        metadata["build_scope"] = {"mode": "training_only", "purpose": "training_robot_holdout_selection_only",
                                   "official_validation_read": False, "official_asset_overlap_checked": False,
                                   "official_validation_windows_created": False, "model_performance_evaluated": False}
    if args.include_unassigned_history:
        metadata["history_policy"] = {"mode": "experimental_unassigned_dynamic_history_only",
            "unassigned_targets_created": False, "missing_label_history_included": False,
            "dynamic_quality_policy": "require_present_finite_float32_raw_fields; no_defaults",
            "target_context_policy": "require_present_finite_float32_and_frozen_categories; no_defaults",
            "unassigned_context_used": False, "policy_selected_as_optimal": False,
            "training_archives": history_sources,
            "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("build_enhanced_dataset.py", "diagnosis_record_quality.py",
                                         "audit_diagnosis_source.py", "audit_diagnosis_continuity.py")}}
    # Publish metadata last so earlier write failures cannot leave usable build metadata.
    with (OUT / "enhanced_meta.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=1)
    if args.training_only:
        print(f"저장 완료: enhanced_train.npz / enhanced_meta.json ({len(ytr)}행); 공식 Validation 미생성", flush=True)
    else:
        print(f"저장 완료: enhanced_* / replay_display.parquet ({len(dd)}행, {dd.robot.nunique()}대)", flush=True)


if __name__ == "__main__":
    main()
