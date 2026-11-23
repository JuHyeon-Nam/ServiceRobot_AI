"""Synthetic timestamp counterfactuals, not model or real-data performance tests."""
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
import audit_diagnosis_history as inspector
from audit_diagnosis_continuity import profile_sequences
from build_enhanced_dataset import WIN, parse_record


def metadata():
    return json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())


def frames(count=65):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return [{"deviceId": "robot-a", "deviceType": "안내로봇",
             "createdAt": (start + pd.Timedelta(seconds=i)).isoformat(),
             "deviceData": {"mainState": "MOVE", "standardLocationData": {"speed": .2}},
             "errorData": {"errorCode": "NOT_ASSIGNED" if i < 5 or i in (20, 21, 31)
                           else "E-RBT-N" if i == 32 else "정상"}} for i in range(count)]


def write_source(root, rows):
    folder = root / "Training/02.라벨링데이터"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "records.zip", "w") as archive:
        for i in reversed(range(len(rows))):
            archive.writestr(f"{i}.json", json.dumps(rows[i]))
    meta = root / "metadata.json"
    meta.write_text(json.dumps(metadata()))
    return meta


def test_paired_counts_match_explicit_history_loops_and_existing_baseline():
    rows = [parse_record(row) for row in frames()]
    before = deepcopy(rows)
    report = inspector.compare_histories({"robot-a": list(reversed(rows))}, metadata(), [1, 2, 90])
    assigned = [r for r in rows if r["errorCode"] != "NOT_ASSIGNED"]
    baseline = profile_sequences({"robot-a": assigned}, metadata(), [1, 2, 90])
    for probe, old in zip(report["probes"], baseline["probes"]):
        expected = {name: dict.fromkeys(("both_retained", "baseline_only_retained",
                                        "history_only_retained", "neither_retained"), 0)
                    for name in metadata()["err_map"]}
        assigned_history = []
        for i, target in enumerate(rows):
            if target["errorCode"] == "NOT_ASSIGNED":
                continue

            def retained(history):
                if len(history) < WIN:
                    return False
                times = [pd.Timestamp(r["createdAt"]) for r in history[-WIN:] + [target]]
                return all((b - a).total_seconds() <= probe["max_gap_seconds"] for a, b in zip(times, times[1:]))

            a, b = retained(assigned_history), retained(rows[:i])
            field = "both_retained" if a and b else "baseline_only_retained" if a else (
                    "history_only_retained" if b else "neither_retained")
            expected[target["errorCode"]][field] += 1
            assigned_history.append(target)
        for name, counts in expected.items():
            actual = probe["per_class"][name]
            assert {field: actual[field] for field in counts} == counts
            assert actual["baseline_retained"] == old["per_class"][name]["retained_windows"]
            assert sum(counts.values()) == report["per_class"][name]["target_records"]
        assert probe["baseline_retained"] == old["retained_windows"]
    rare = report["probes"][0]["per_class"]["E-RBT-N"]
    assert rare["baseline_retained"] == 0 and rare["inclusive_retained"] == 1
    assert report["per_class"]["E-RBT-N"]["unassigned_in_history_targets"] == 1
    assert "NOT_ASSIGNED" not in report["per_class"]
    assert rows == before


def test_target_gap_is_not_bridged_by_an_earlier_unassigned_record():
    rows = frames(count=33)
    rows[32]["createdAt"] = "2026-01-01T00:02:01Z"
    report = inspector.compare_histories({"robot-a": [parse_record(r) for r in rows]}, metadata(), [1, 90])
    assert [p["per_class"]["E-RBT-N"]["inclusive_retained"] for p in report["probes"]] == [0, 1]


@pytest.mark.parametrize("count", [1, 30, 31])
def test_unassigned_records_never_become_targets_or_normal_labels(count):
    rows = frames(count)
    for row in rows:
        row["errorData"]["errorCode"] = "NOT_ASSIGNED"
    result = inspector.compare_histories({"robot-a": [parse_record(r) for r in rows]}, metadata(), [1])
    assert result["probes"][0]["inclusive_retained"] == 0
    assert all(row["target_records"] == 0 for row in result["per_class"].values())


@pytest.mark.parametrize("issue", ["duplicate", "mixed", "missing"])
def test_unassigned_history_clock_is_validated(issue):
    rows = frames()
    rows[0]["createdAt"] = {"duplicate": rows[1]["createdAt"], "mixed": "2026-01-01T00:00:00",
                             "missing": None}[issue]
    with pytest.raises(ValueError):
        inspector.compare_histories({"robot-a": [parse_record(r) for r in rows]}, metadata(), [1])


@pytest.mark.parametrize("gaps", [[], [True], [float("nan")], [1, 1]])
def test_invalid_probes_fail_before_source_io(tmp_path, monkeypatch, gaps):
    monkeypatch.setattr(inspector, "load_encoding_metadata", lambda *a: pytest.fail("validate probes first"))
    with pytest.raises(ValueError):
        inspector.audit_history(tmp_path, tmp_path / "absent", gaps)


def test_cli_preserves_sources_excludes_missing_labels_and_never_reads_validation(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    rows = frames()
    rows.append({"errorData": {"errorCode": None}})
    meta = write_source(root, rows)
    bad = root / "Validation/02.라벨링데이터/bad.zip"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"invalid official Validation")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(np, "savez_compressed", lambda *a, **k: pytest.fail("no sensor dataset creation"))
    original = inspector.read_histories

    def tracked(folder, *args):
        assert folder == root / "Training/02.라벨링데이터"
        return original(folder, *args)

    monkeypatch.setattr(inspector, "read_histories", tracked)
    output = tmp_path / "report.json"
    args = ["--base-dir", str(root), "--encoding-meta", str(meta), "--report", str(output)]
    assert inspector.main(args) == 0
    report = json.loads(output.read_text())
    assert report["ingestion"] == {"json_records": 66, "missing_label_excluded": 1,
                                   "unassigned_history_records": 8, "assigned_target_records": 57,
                                   "non_json_members": 0}
    assert not report["official_validation_read"] and not report["unassigned_used_as_target"]
    assert not report["model_trained"] and not report["sensor_window_arrays_created"]
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        inspector.main(args)
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(SystemExit):
        inspector.main([*args[:-1], str(link)])
    assert output.read_bytes() == saved and not (tmp_path / "absent").exists()
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("issue", ["malformed", "category", "asset"])
def test_invalid_records_abort_without_a_report(tmp_path, issue):
    root = tmp_path / "raw"
    rows = frames()
    if issue == "category":
        rows[32]["errorData"]["errorCode"] = "UNKNOWN"
    elif issue == "asset":
        rows[0]["deviceId"] = " robot-a "
    meta = write_source(root, rows)
    if issue == "malformed":
        with zipfile.ZipFile(root / "Training/02.라벨링데이터/records.zip", "a") as archive:
            archive.writestr("bad.json", b"{")
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        inspector.main(["--base-dir", str(root), "--encoding-meta", str(meta), "--report", str(output)])
    assert not output.exists()


def test_source_mutation_during_history_scan_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    meta = write_source(root, frames())
    original = inspector.read_histories

    def mutate(folder, *args):
        result = original(folder, *args)
        with (folder / "records.zip").open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(inspector, "read_histories", mutate)
    with pytest.raises(ValueError, match="changed during"):
        inspector.audit_history(root, meta)
