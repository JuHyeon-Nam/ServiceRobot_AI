"""Opt-in raw-history preprocessing on synthetic ZIPs; no model-performance claims."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_enhanced_dataset as builder
from audit_diagnosis_history import compare_histories
from train_diagnosis import load_dataset


def metadata():
    return json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())


def frames():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(66):
        rows.append({"deviceId": "robot-a", "deviceType": "안내로봇",
            "createdAt": (start + pd.Timedelta(seconds=i + (89 if i >= 32 else 0))).isoformat(),
            "deviceData": {"mainState": "MOVE", "batteryLevel": i, "collision": False, "obstacle": 0,
                "isOffline": 0, "nowCharging": False, "emergencyStop": False,
                "standardLocationData": {"x": i, "y": 0, "speed": i / 100, "degree": 0},
                "totalOperationData": {"batteryUse": i, "batteryCycleCount": 0, "distance": i * 2}},
            "siteData": {"crowd": "MIDDLE"}, "errorData": {
                "errorCode": "NOT_ASSIGNED" if i in (0, 5, 31, 34) else "E-RBT-N" if i == 32 else "정상"}})
    return rows


def source(root, rows):
    folder = root / "Training/02.라벨링데이터"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "records.zip", "w") as archive:
        for i in reversed(range(len(rows))):
            archive.writestr(f"{i}.json", json.dumps(rows[i]))
    meta = root / "frozen.json"
    meta.write_text(json.dumps(metadata()))
    return meta


def arguments(root, meta, output, gap=90):
    return ["--base-dir", str(root), "--encoding-meta", str(meta), "--out-dir", str(output),
            "--training-only", "--reject-malformed-records", "--max-gap-seconds", str(gap),
            "--include-unassigned-history"]


@pytest.mark.parametrize("gap", [1, 90])
def test_actual_history_arrays_context_targets_and_counterfactual_counts_agree(tmp_path, monkeypatch, gap):
    root, output, rows = tmp_path / "raw", tmp_path / "built", frames()
    # Unused context is intentionally not mapped or fabricated into target S.
    for i in (0, 31):
        rows[i]["deviceData"]["mainState"] = ["unused-history-state"]
        rows[i]["deviceType"] = ["unused-history-type"]
        rows[i]["deviceData"]["totalOperationData"]["distance"] = float("nan")
    meta = source(root, rows)
    bad = root / "Validation/02.라벨링데이터/bad.zip"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"official Validation must not be read")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    reads, original = [], builder.read_split

    def tracked(folder, tag, **kwargs):
        assert tag == "train"
        reads.append(tag)
        return original(folder, tag, **kwargs)

    monkeypatch.setattr(builder, "read_split", tracked)
    builder.main(arguments(root, meta, output, gap))
    built = json.loads((output / "enhanced_meta.json").read_text())
    parsed = [builder.parse_record(row) for row in rows]
    profile = compare_histories({"robot-a": parsed}, metadata(), [gap])["probes"][0]
    loaded = load_dataset(output / "enhanced_train.npz")
    assert reads == ["train"] and loaded["features"].shape[1] == 249
    assert set(p.name for p in output.iterdir()) == {"enhanced_train.npz", "enhanced_meta.json"}
    assert built["err_map"] == metadata()["err_map"] and "NOT_ASSIGNED" not in built["err_map"]
    assert not built["history_policy"]["unassigned_context_used"]
    audit = built["sequence_audit"]["splits"]["train"]
    assert audit["emitted_windows"] == profile["inclusive_retained"]
    assert audit["candidate_windows"] == (audit["unassigned_target_windows"] + audit["discarded_gap_windows"] + audit["emitted_windows"])
    assert audit["labeled_candidate_windows"] == audit["discarded_gap_windows"] + audit["emitted_windows"]
    assert audit["emitted_windows_with_unassigned_history"] > 0
    with np.load(output / "enhanced_train.npz") as arrays:
        times = [row["createdAt"] for row in rows]
        for index, stamp in enumerate(arrays["target_times"]):
            target = times.index(stamp)
            assert rows[target]["errorData"]["errorCode"] != "NOT_ASSIGNED"
            expected = [[row[c] for c in builder.DYN] for row in parsed[target - 30:target]]
            np.testing.assert_array_equal(arrays["X"][index], np.asarray(expected, dtype=np.float32))
            np.testing.assert_array_equal(arrays["S"][index, :7], [parsed[target][c] for c in builder.STAT_NUM])
            assert arrays["y"][index] == built["err_map"][rows[target]["errorData"]["errorCode"]]
        assert np.isfinite(arrays["X"]).all() and np.isfinite(arrays["S"]).all()
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("extras", [[], ["--training-only", "--encoding-meta", "absent"],
    ["--training-only", "--encoding-meta", "absent", "--max-gap-seconds", "10"],
    ["--training-only", "--reject-malformed-records", "--max-gap-seconds", "10"],
    ["--encoding-meta", "absent", "--reject-malformed-records", "--max-gap-seconds", "10"]])
def test_experimental_mode_requires_explicit_scope_encoding_gap_and_strictness(tmp_path, monkeypatch, extras):
    monkeypatch.setattr(builder, "read_split", lambda *a, **k: pytest.fail("reject options before input"))
    monkeypatch.setattr(builder, "load_encoding_metadata", lambda *a: pytest.fail("reject options before metadata"))
    with pytest.raises(SystemExit):
        builder.main(["--base-dir", str(tmp_path), "--include-unassigned-history", *extras])


@pytest.mark.parametrize("issue", ["missing_dynamic", "null_dynamic", "nonfinite_dynamic", "huge_dynamic",
                                    "target_context", "target_category", "target_crowd"])
def test_bad_used_values_fail_before_output_without_fallbacks(tmp_path, issue):
    root, output, rows = tmp_path / "raw", tmp_path / "built", frames()
    if issue == "missing_dynamic":
        rows[0]["deviceData"].pop("batteryLevel")
    elif issue == "null_dynamic":
        rows[0]["deviceData"]["batteryLevel"] = None
    elif issue == "nonfinite_dynamic":
        rows[0]["deviceData"]["batteryLevel"] = float("nan")
    elif issue == "huge_dynamic":
        rows[0]["deviceData"]["batteryLevel"] = 1e40
    elif issue == "target_context":
        rows[32]["deviceData"]["totalOperationData"].pop("distance")
    elif issue == "target_category":
        rows[32]["deviceData"]["mainState"] = "UNKNOWN"
    else:
        rows[32]["siteData"]["crowd"] = "UNKNOWN"
    meta = source(root, rows)
    with pytest.raises(ValueError, match="quality rejected"):
        builder.main(arguments(root, meta, output))
    assert not output.exists()


@pytest.mark.parametrize("missing", ["training", "encoding", "strict"])
def test_read_split_cannot_enable_history_in_an_unprotected_scope(tmp_path, missing):
    options = {"include_unassigned_history": True, "encoding_metadata": metadata(), "reject_malformed_records": True}
    tag = "validation" if missing == "training" else "train"
    if missing == "encoding":
        options["encoding_metadata"] = None
    if missing == "strict":
        options["reject_malformed_records"] = False
    with pytest.raises(ValueError, match="strict Training"):
        builder.read_split(tmp_path, tag, **options)


def test_default_history_filter_and_source_mutation_guards_remain(tmp_path, monkeypatch):
    root, output, rows = tmp_path / "raw", tmp_path / "built", frames()
    meta = source(root, rows)
    default, _, _ = builder.read_split(root / "Training/02.라벨링데이터", "train")
    assert all(r["errorCode"] != "NOT_ASSIGNED" for r in default["robot-a"])
    original = builder.read_split

    def mutate(folder, *args, **kwargs):
        result = original(folder, *args, **kwargs)
        with (Path(folder) / "records.zip").open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(builder, "read_split", mutate)
    with pytest.raises(ValueError, match="changed during"):
        builder.main(arguments(root, meta, output))
    assert not output.exists()
