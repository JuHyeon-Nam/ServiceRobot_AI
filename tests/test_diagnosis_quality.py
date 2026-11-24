"""Raw field checks on synthetic JSON; never model performance or physical calibration."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import audit_diagnosis_quality as inspector
from build_enhanced_dataset import parse_record
from diagnosis_record_quality import MISSING, numeric_issue, record_quality


def metadata():
    return json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())


def frame(code="정상", second=0):
    return {"deviceId": "robot-a", "deviceType": "안내로봇", "createdAt": f"2026-01-01T00:00:{second:02d}Z",
            "deviceData": {"mainState": "MOVE", "batteryLevel": 0, "collision": False, "obstacle": 0,
                "isOffline": 0, "nowCharging": False, "emergencyStop": False,
                "standardLocationData": {"x": 0, "y": 0, "speed": 0, "degree": 0},
                "totalOperationData": {"batteryUse": 0, "batteryCycleCount": 0, "distance": 0}},
            "siteData": {"crowd": "MIDDLE"}, "errorData": {"errorCode": code}}


def write_source(root, rows):
    folder = root / "Training/02.라벨링데이터"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "records.zip", "w") as archive:
        for index, row in enumerate(rows):
            archive.writestr(f"{index}.json", json.dumps(row))
    meta = root / "frozen.json"
    meta.write_text(json.dumps(metadata()))
    return meta


@pytest.mark.parametrize("value,reason", [(MISSING, "missing_defaults_to_zero"),
    (None, "null_defaults_to_zero"), ("bad", "invalid_numeric_defaults_to_zero"),
    ([], "invalid_numeric_defaults_to_zero"), (float("nan"), "nonfinite"),
    (float("inf"), "nonfinite"), (-float("inf"), "nonfinite"),
    (1e40, "outside_float32"), (10**1000, "outside_float32"),
    (0, None), (False, None), (True, None), ("0.25", None), (-2.5, None)])
def test_numeric_missing_invalid_and_real_zero_are_distinguished(value, reason):
    assert numeric_issue(value) == reason


def test_history_dynamic_and_target_context_requirements_are_separate_and_pure():
    document = frame("NOT_ASSIGNED")
    document["deviceData"]["totalOperationData"].pop("distance")
    document["deviceData"]["mainState"] = "UNKNOWN"
    document["siteData"].pop("crowd")
    before = deepcopy(document)
    result = record_quality(document, parse_record(document), metadata())
    assert result["dynamic_usable"] and not result["target_context_usable"]
    assert result["context_issues"] == {"distance": "missing_defaults_to_zero",
        "mainState": "missing_or_unknown_frozen_category", "crowd": "missing_or_unknown_defaults_to_middle"}
    document["deviceData"]["batteryLevel"] = None
    result = record_quality(document, parse_record(document), metadata())
    assert result["dynamic_issues"] == {"batteryLevel": "null_defaults_to_zero"}
    document["deviceData"]["batteryLevel"] = 0
    assert document == before


def test_audit_partitions_records_reports_defaults_and_does_not_read_validation(tmp_path):
    root = tmp_path / "raw"
    missing_context = frame("NOT_ASSIGNED", 1)
    missing_context["deviceData"]["totalOperationData"].pop("distance")
    bad_sensor = frame("NOT_ASSIGNED", 2)
    bad_sensor["deviceData"]["standardLocationData"]["speed"] = "bad"
    missing_label = frame(None, 3)
    unknown_label = frame("UNKNOWN", 4)
    meta = write_source(root, [frame(), missing_context, bad_sensor, missing_label, unknown_label, []])
    bad = root / "Validation/02.라벨링데이터/bad.zip"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a valid official archive")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = inspector.audit_quality(root, meta)
    assert report["json_records"] == 6 and report["malformed_records"] == {"AttributeError": 1}
    assigned, unassigned, missing = [report["buckets"][key] for key in ("assigned", "unassigned", "missing_label")]
    assert assigned["records"] == assigned["all_inputs_usable_records"] == 2
    assert assigned["unknown_target_label_records"] == 1
    assert unassigned["records"] == 2 and unassigned["dynamic_usable_records"] == 1
    assert unassigned["target_context_usable_records"] == 1 and unassigned["all_inputs_usable_records"] == 0
    assert unassigned["dynamic_issues"] == {"speed:invalid_numeric_defaults_to_zero": 1}
    assert missing["records"] == 1 and report["clock"]["timestamped_records"] == 4
    assert not report["official_validation_read"] and not report["model_trained"]
    assert report["training_readiness"] == "not_evaluated"
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("issue", ["duplicate", "mixed", "timestamp", "asset"])
def test_invalid_clocks_and_assets_are_reported_not_hidden(tmp_path, issue):
    rows = [frame(), frame("NOT_ASSIGNED", 1)]
    if issue == "duplicate":
        rows[1]["createdAt"] = rows[0]["createdAt"]
    elif issue == "mixed":
        rows[1]["createdAt"] = "2026-01-01T00:00:01"
    elif issue == "timestamp":
        rows[1]["createdAt"] = "bad"
    else:
        rows[1]["deviceId"] = "nan"
    root = tmp_path / "raw"
    report = inspector.audit_quality(root, write_source(root, rows))
    key = {"duplicate": "duplicate_timestamp_groups", "mixed": "mixed_clock_assets",
           "timestamp": "invalid_timestamp_records", "asset": "invalid_asset_records"}[issue]
    assert report["clock"][key]


def test_cli_report_protection_and_source_mutation_detection(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    meta = write_source(root, [frame()])
    output = tmp_path / "report.json"
    args = ["--base-dir", str(root), "--encoding-meta", str(meta), "--report", str(output)]
    assert inspector.main(args) == 0
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        inspector.main(args)
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(SystemExit):
        inspector.main([*args[:-1], str(link)])
    original = inspector.record_quality

    def mutate(*args):
        with (root / "Training/02.라벨링데이터/records.zip").open("ab") as stream:
            stream.write(b"changed")
        return original(*args)

    monkeypatch.setattr(inspector, "record_quality", mutate)
    with pytest.raises(ValueError, match="changed during"):
        inspector.audit_quality(root, meta)
    assert output.read_bytes() == saved and not (tmp_path / "absent").exists()


def test_json_budget_and_archive_failures_are_not_silently_skipped(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    meta = write_source(root, [frame()])
    monkeypatch.setattr(inspector, "MAX_JSON_BYTES", 3)
    with pytest.raises(ValueError, match="budget"):
        inspector.audit_quality(root, meta)
    (root / "Training/02.라벨링데이터/records.zip").write_bytes(b"corrupt")
    with pytest.raises(zipfile.BadZipFile):
        inspector.audit_quality(root, meta)
