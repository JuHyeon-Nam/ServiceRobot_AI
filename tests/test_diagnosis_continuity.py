"""Temporal attrition checks on synthetic records/ZIPs, not model performance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import audit_diagnosis_continuity as inspector
import build_enhanced_dataset as builder


def metadata():
    return json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())


def frames(asset="a", count=62):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return [{"deviceId": asset, "deviceType": "안내로봇",
             "createdAt": (start + pd.Timedelta(seconds=i + (89 if i >= 31 else 0))).isoformat(),
             "deviceData": {"mainState": "MOVE", "batteryLevel": 90, "standardLocationData": {"speed": .2}},
             "errorData": {"errorCode": "E-RBT-N" if i == 31 else "정상"}} for i in range(count)]


def parsed(rows):
    return [builder.parse_record(row) for row in rows]


def write_source(root, rows):
    folder = root / "Training/02.라벨링데이터"
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / "records.zip", "w") as archive:
        for i in reversed(range(len(rows))):
            archive.writestr(f"{i}.json", json.dumps(rows[i]))
    path = root / "frozen.json"
    path.write_text(json.dumps(metadata()))
    return path


def test_target_edge_class_loss_and_exact_gap_boundary_preserve_inputs():
    by_asset = {"a": list(reversed(parsed(frames())))}
    before = deepcopy(by_asset)
    report = inspector.profile_sequences(by_asset, metadata(), [1, 90])
    assert report["candidate_windows"] == 32
    assert report["baseline_support"]["E-RBT-N"] == {"windows": 1, "assets": 1}
    strict, boundary = report["probes"]
    assert strict["gap_edges"] == 1 and strict["retained_windows"] == 2
    assert strict["discarded_windows"] == 30
    assert strict["per_class"]["E-RBT-N"] == {"retained_windows": 0, "discarded_windows": 1, "retained_assets": 0}
    assert boundary["retained_windows"] == 32 and boundary["gap_edges"] == 0
    assert report["adjacent_intervals"] == 61
    assert report["interval_quantiles_ns"]["max"] == 90 * 10**9
    assert report["max_window_span_ns"] == 119 * 10**9
    assert by_asset == before


def test_nanosecond_precision_includes_the_target_observation():
    rows = frames(count=31)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for i, row in enumerate(rows):
        row["createdAt"] = (start + pd.Timedelta(nanoseconds=i + int(i == 30))).isoformat()
    report = inspector.profile_sequences({"a": parsed(rows)}, metadata(), [1e-9, 2e-9])
    assert [p["retained_windows"] for p in report["probes"]] == [0, 1]
    assert report["max_window_span_ns"] == 31


def test_short_assets_and_no_intervals_are_not_reported_as_usable_windows():
    report = inspector.profile_sequences({"a": parsed(frames(count=1))}, metadata(), [1])
    assert report["candidate_windows"] == 0 and report["max_window_span_ns"] is None
    assert all(value is None for value in report["interval_quantiles_ns"].values())
    assert report["probes"][0]["retention_fraction"] is None
    assert all(row["windows"] == row["assets"] == 0 for row in report["baseline_support"].values())


@pytest.mark.parametrize("gaps", [[], [None], [0], [-1], [True], [float("nan")], [float("inf")], [1e-12], [1e30], [1, 1]])
def test_invalid_or_duplicate_probes_rejected_before_input(gaps, tmp_path, monkeypatch):
    monkeypatch.setattr(inspector, "load_encoding_metadata", lambda *a: pytest.fail("reject invalid probe before metadata"))
    with pytest.raises((ValueError, TypeError)):
        inspector.audit_continuity(tmp_path, tmp_path / "absent", gaps)


@pytest.mark.parametrize("issue", ["duplicate", "mixed", "missing", "unknown"])
def test_existing_clock_and_encoding_guards_are_preserved(issue):
    rows = frames()
    if issue == "duplicate":
        rows[10]["createdAt"] = rows[9]["createdAt"]
    elif issue == "mixed":
        rows[10]["createdAt"] = "2026-01-01T00:00:10"
    elif issue == "missing":
        rows[10]["createdAt"] = None
    else:
        rows[10]["errorData"]["errorCode"] = "UNKNOWN"
    with pytest.raises(ValueError):
        inspector.profile_sequences({"a": parsed(rows)}, metadata(), [1])


@pytest.mark.parametrize("gap", [1, 90])
def test_probe_counts_match_actual_training_only_builder(tmp_path, gap):
    root = tmp_path / "raw"
    rows = frames() + frames("b", count=35)
    meta = write_source(root, rows)
    report = inspector.audit_continuity(root, meta, [gap])
    output = tmp_path / "built"
    builder.main(["--base-dir", str(root), "--encoding-meta", str(meta), "--out-dir", str(output),
                  "--training-only", "--max-gap-seconds", str(gap)])
    built_meta = json.loads((output / "enhanced_meta.json").read_text())
    with np.load(output / "enhanced_train.npz") as arrays:
        probe = report["probes"][0]
        assert len(arrays["y"]) == probe["retained_windows"]
        for name, code in metadata()["err_map"].items():
            mask = arrays["y"] == code
            assert probe["per_class"][name]["retained_windows"] == int(mask.sum())
            assert probe["per_class"][name]["retained_assets"] == len(np.unique(arrays["groups"][mask]))
    assert probe["discarded_windows"] == built_meta["sequence_audit"]["splits"]["train"]["discarded_gap_windows"]


def test_cli_never_reads_validation_or_creates_sensor_arrays_and_protects_output(tmp_path, monkeypatch, capsys):
    root = tmp_path / "raw"
    meta = write_source(root, frames())
    bad = root / "Validation/02.라벨링데이터/corrupt.zip"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"unusable official Validation")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    loaded, original = [], inspector.read_split

    def track(folder, tag, **kwargs):
        assert Path(folder) == root / "Training/02.라벨링데이터"
        loaded.append(tag)
        return original(folder, tag, **kwargs)

    monkeypatch.setattr(inspector, "read_split", track)
    monkeypatch.setattr(np, "savez_compressed", lambda *a, **k: pytest.fail("audit must not create window arrays"))
    output = tmp_path / "report.json"
    args = ["--base-dir", str(root), "--encoding-meta", str(meta), "--report", str(output)]
    assert inspector.main(args) == 0
    result = json.loads(output.read_text())
    assert loaded == ["train"] and not result["official_validation_read"]
    assert not result["model_trained"] and not result["policy_selected"]
    assert not result["sensor_window_arrays_created"]
    assert result["ingestion"]["accepted_records"] == 62
    assert {p: p.read_bytes() for p in before} == before
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        inspector.main(args)
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(SystemExit):
        inspector.main([*args[:-1], str(link)])
    assert output.read_bytes() == saved and not (tmp_path / "absent").exists()


def test_source_change_during_scan_rejected(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    meta = write_source(root, frames())
    original = inspector.read_split

    def mutate(folder, *args, **kwargs):
        result = original(folder, *args, **kwargs)
        with (Path(folder) / "records.zip").open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(inspector, "read_split", mutate)
    with pytest.raises(ValueError, match="changed during"):
        inspector.audit_continuity(root, meta)
