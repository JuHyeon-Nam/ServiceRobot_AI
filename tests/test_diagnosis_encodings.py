"""Frozen encoding contracts on synthetic ZIPs, not real-data retraining evidence."""
import hashlib
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
from train_diagnosis import load_dataset, validate_label_coverage, validate_metadata_contract

SERVING = ROOT / "data/processed"


def records(asset, device_type, state, fault, count=35):
    return [{"deviceId": asset, "deviceType": device_type, "createdAt": f"2026-01-01T00:00:{i:02d}Z",
             "errorData": {"errorCode": fault if i % 2 else "정상"},
             "deviceData": {"mainState": state, "batteryLevel": 90 - i,
                            "standardLocationData": {"x": i, "y": i / 10, "speed": .2}}}
            for i in range(count)]


def write_split(base, split, rows):
    folder = base / split / "02.라벨링데이터"
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / "records.zip", "w") as archive:
        for i in reversed(range(len(rows))):
            archive.writestr(f"{i}.json", json.dumps(rows[i]))


@pytest.fixture
def encoding_inputs(tmp_path):
    base, source = tmp_path / "raw", tmp_path / "encoding.json"
    source.write_bytes((SERVING / "enhanced_meta.json").read_bytes())
    train = records("train-a", "안내로봇", "MOVE", "E-RBT-B")
    validation = records("val-a", "청소로봇", "OFFLINE", "E-ENV-O")
    write_split(base, "Training", train)
    write_split(base, "Validation", validation)
    return base, source, train, validation


def test_frozen_maps_keep_absent_categories_and_validation_only_known_codes(encoding_inputs, tmp_path, monkeypatch):
    base, source, _, _ = encoding_inputs
    source_bytes = source.read_bytes()
    metadata = json.loads(source_bytes)
    archives = {path: path.read_bytes() for path in base.rglob("*.zip")}
    output = tmp_path / "frozen"
    reads, original_read = [], Path.read_bytes

    def track_source(path):
        if path == source:
            reads.append(path)
        return original_read(path)

    with monkeypatch.context() as tracked:
        tracked.setattr(Path, "read_bytes", track_source)
        builder.main(["--base-dir", str(base), "--out-dir", str(output), "--encoding-meta", str(source)])
    assert reads == [source]
    built = json.loads((output / "enhanced_meta.json").read_text())
    for key in ("err_map", "devtype_map", "mainstate_map", "dyn", "stat", "crowd_map"):
        assert built[key] == metadata[key]
    assert built["encoding_audit"] == {
        "mode": "frozen_metadata", "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "unknown_category_policy": "reject_deviceType_mainState_errorCode"}
    names = validate_metadata_contract(built, metadata, json.loads((SERVING / "robot_pdm_enhanced_meta.json").read_text()))
    for split, asset, device_type, state, fault in (("train", "train-a", "안내로봇", "MOVE", "E-RBT-B"),
                                                  ("val", "val-a", "청소로봇", "OFFLINE", "E-ENV-O")):
        loaded = load_dataset(output / f"enhanced_{split}.npz")
        assert loaded["features"].shape == (5, 249)
        assert loaded["groups"].tolist() == [asset] * 5
        assert loaded["y"].tolist() == [metadata["err_map"][fault if i % 2 else "정상"] for i in range(30, 35)]
        np.testing.assert_array_equal(loaded["S"][:, -2], [metadata["devtype_map"][device_type]] * 5)
        np.testing.assert_array_equal(loaded["S"][:, -1], [metadata["mainstate_map"][state]] * 5)
        # Correct encodings do not make this small, class-incomplete fixture training-ready.
        with pytest.raises(ValueError, match="missing declared diagnosis classes"):
            validate_label_coverage(loaded["y"], names, split)
    display = pd.read_parquet(output / "replay_display.parquet")
    assert display.validx.tolist() == list(range(5)) and display.x.tolist() == list(range(30, 35))
    assert display.robot.tolist() == ["val-a"] * 5
    assert source.read_bytes() == source_bytes
    assert {path: path.read_bytes() for path in archives} == archives


def test_default_union_mapping_remains_available_without_reusing_source_maps(encoding_inputs, tmp_path):
    base, _, _, _ = encoding_inputs
    output = tmp_path / "legacy"
    builder.main(["--base-dir", str(base), "--out-dir", str(output)])
    metadata = json.loads((output / "enhanced_meta.json").read_text())
    assert metadata["devtype_map"] == {"안내로봇": 0, "청소로봇": 1}
    assert metadata["mainstate_map"] == {"MOVE": 0, "OFFLINE": 1}
    assert metadata["err_map"] == {"E-ENV-O": 0, "E-RBT-B": 1, "정상": 2}
    assert metadata["encoding_audit"] == {"mode": "fit_on_official_split_union", "source_path": None,
                                          "source_sha256": None, "unknown_category_policy": "legacy_defaults"}


@pytest.mark.parametrize("split", ["Training", "Validation"])
@pytest.mark.parametrize("field", ["deviceType", "mainState", "errorCode"])
def test_unknown_categories_are_rejected_in_either_split_before_output(encoding_inputs, tmp_path, split, field):
    base, source, train, validation = encoding_inputs
    rows = train if split == "Training" else validation
    if field == "mainState":
        rows[0]["deviceData"][field] = "UNREGISTERED"
    elif field == "errorCode":
        rows[0]["errorData"][field] = "UNREGISTERED"
    else:
        rows[0][field] = "UNREGISTERED"
    write_split(base, split, rows)
    before = {path: path.read_bytes() for path in base.rglob("*.zip")}
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=f"unknown {field}") as error:
        builder.main(["--base-dir", str(base), "--out-dir", str(output), "--encoding-meta", str(source)])
    assert ("train" if split == "Training" else "validation") in str(error.value)
    assert not output.exists()
    assert {path: path.read_bytes() for path in before} == before


def test_unknown_in_short_history_is_not_silently_ignored(encoding_inputs, tmp_path):
    base, source, train, _ = encoding_inputs
    write_split(base, "Training", train + records("short-a", "UNKNOWN", "MOVE", "E-RBT-B", count=1))
    output = tmp_path / "rejected-short"
    with pytest.raises(ValueError, match="unknown deviceType.*short-a"):
        builder.main(["--base-dir", str(base), "--out-dir", str(output), "--encoding-meta", str(source)])
    assert not output.exists()


@pytest.mark.parametrize("key,value", [
    ("dyn", []), ("stat", []), ("crowd_map", {"LOW": 0, "MIDDLE": 0, "HIGH": 2}),
    ("crowd_map", {"LOW": False, "MIDDLE": True, "HIGH": 2}),
    ("err_map", {"fault": 0}), ("err_map", {"정상": 0, "fault": 0}),
    ("devtype_map", {}), ("devtype_map", {" robot": 0}), ("devtype_map", {"robot": "0"}),
    ("mainstate_map", {"MOVE": True}), ("mainstate_map", {"MOVE": 1}),
])
def test_invalid_frozen_metadata_fails_before_zip_io(tmp_path, monkeypatch, key, value):
    metadata = json.loads((SERVING / "enhanced_meta.json").read_text())
    metadata[key] = value
    source, output = tmp_path / "bad.json", tmp_path / "output"
    source.write_text(json.dumps(metadata))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid frozen metadata must be rejected before any ZIP is read")

    monkeypatch.setattr(builder, "read_split", forbidden)
    with pytest.raises(ValueError, match="encoding metadata"):
        builder.main(["--base-dir", str(tmp_path), "--out-dir", str(output), "--encoding-meta", str(source)])
    assert not output.exists()


@pytest.mark.parametrize("content", [b"[]", b"{"])
def test_non_object_or_malformed_source_metadata_is_not_a_record_skip(tmp_path, content):
    source = tmp_path / "bad.json"
    source.write_bytes(content)
    with pytest.raises(ValueError):
        builder.load_encoding_metadata(source)


def test_frozen_encoding_does_not_redefine_sensor_or_crowd_fallbacks(encoding_inputs, tmp_path):
    base, source, train, _ = encoding_inputs
    train[0]["deviceData"]["batteryLevel"] = "not-a-number"
    train[30]["siteData"] = {"crowd": "UNREGISTERED"}
    write_split(base, "Training", train)
    output = tmp_path / "fallback"
    builder.main(["--base-dir", str(base), "--out-dir", str(output), "--encoding-meta", str(source),
                  "--reject-malformed-records"])
    loaded = load_dataset(output / "enhanced_train.npz")
    assert loaded["X"][0, 0, 0] == 0
    assert loaded["S"][0, 6] == 1
