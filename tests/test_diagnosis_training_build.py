"""Training-only preprocessing on synthetic ZIPs; not real-data model scores."""
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
import train_diagnosis as trainer


def frames(asset, count=62, gap=False, all_classes=False):
    codes = list(json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())["err_map"])
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return [{"deviceId": asset, "deviceType": "안내로봇",
             "createdAt": (start + pd.Timedelta(seconds=i + (89 if gap and i >= 31 else 0))).isoformat(),
             "deviceData": {"mainState": "MOVE", "batteryLevel": 90 - .1 * i,
                            "standardLocationData": {"x": i, "y": i / 10, "speed": .2}},
             "errorData": {"errorCode": codes[i % len(codes)] if all_classes else "정상"}}
            for i in range(count)]


def write_split(base, split, rows):
    folder = base / split / "02.라벨링데이터"
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / "records.zip", "w") as archive:
        for i in reversed(range(len(rows))):
            row = rows[i]
            archive.writestr(f"{i}.json", row if isinstance(row, bytes) else json.dumps(row))


@pytest.fixture
def inputs(tmp_path):
    base, metadata = tmp_path / "raw", tmp_path / "encoding.json"
    metadata.write_bytes((ROOT / "data/processed/enhanced_meta.json").read_bytes())
    write_split(base, "Training", frames("train-a", gap=True))
    return base, metadata


def arguments(base, metadata, output, *extras):
    return ["--base-dir", str(base), "--out-dir", str(output), "--encoding-meta", str(metadata),
            "--training-only", *extras]


@pytest.mark.parametrize("validation", ["missing", "corrupt", "overlapping"])
def test_only_training_is_read_and_no_validation_or_display_is_created(inputs, tmp_path, monkeypatch, validation):
    base, metadata = inputs
    if validation == "corrupt":
        folder = base / "Validation/02.라벨링데이터"
        folder.mkdir(parents=True)
        (folder / "corrupt.zip").write_bytes(b"invalid ZIP")
    elif validation == "overlapping":
        write_split(base, "Validation", frames("train-a"))
    before = {p: p.read_bytes() for p in base.rglob("*.zip")}
    metadata_before = metadata.read_bytes()
    reads, original = [], builder.read_split

    def track(folder, tag, **kwargs):
        assert Path(folder) == base / "Training/02.라벨링데이터"
        reads.append(tag)
        return original(folder, tag, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("training-only build must not create display or read official Validation")

    monkeypatch.setattr(builder, "read_split", track)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", forbidden)
    output = tmp_path / "training"
    builder.main(arguments(base, metadata, output, "--max-gap-seconds", "1", "--reject-malformed-records"))
    assert reads == ["train"]
    assert {p.name for p in output.iterdir()} == {"enhanced_train.npz", "enhanced_meta.json"}
    built = json.loads((output / "enhanced_meta.json").read_text())
    source = json.loads(metadata_before)
    for field in ("err_map", "devtype_map", "mainstate_map", "dyn", "stat", "crowd_map"):
        assert built[field] == source[field]
    assert built["build_scope"] == {"mode": "training_only", "purpose": "training_robot_holdout_selection_only",
        "official_validation_read": False, "official_asset_overlap_checked": False,
        "official_validation_windows_created": False, "model_performance_evaluated": False}
    assert built["asset_audit"] == {"train": ["train-a"], "validation": None, "overlap": None}
    assert set(built["ingestion_audit"]["splits"]) == set(built["sequence_audit"]["splits"]) == {"train"}
    assert built["encoding_audit"]["source_sha256"] == hashlib.sha256(metadata_before).hexdigest()
    sequence = built["sequence_audit"]["splits"]["train"]
    assert sequence["gaps"] == 1 and sequence["discarded_gap_windows"] == 30
    with np.load(output / "enhanced_train.npz", allow_pickle=False) as rows:
        assert rows["X"].shape == (2, 30, 7) and rows["S"].shape == (2, 9)
        assert rows["groups"].tolist() == ["train-a"] * 2
        assert rows["target_times"].tolist() == [frames("train-a", gap=True)[i]["createdAt"] for i in (30, 61)]
        np.testing.assert_array_equal(rows["X"][:, 0, 2], [0, 31])
        np.testing.assert_array_equal(rows["X"][:, -1, 2], [29, 60])
        assert rows["y"].tolist() == [source["err_map"]["정상"]] * 2
    assert trainer.load_dataset(output / "enhanced_train.npz")["features"].shape == (2, 249)
    assert {p: p.read_bytes() for p in before} == before
    assert metadata.read_bytes() == metadata_before


@pytest.mark.parametrize("gap", [None, "1"])
def test_training_arrays_and_audits_equal_the_unchanged_full_build(inputs, tmp_path, gap):
    base, metadata = inputs
    write_split(base, "Validation", frames("validation-a", gap=True))
    only, full = tmp_path / "only", tmp_path / "full"
    extra = [] if gap is None else ["--max-gap-seconds", gap]
    builder.main(arguments(base, metadata, only, *extra))
    builder.main(["--base-dir", str(base), "--encoding-meta", str(metadata), "--out-dir", str(full), *extra])
    with np.load(only / "enhanced_train.npz") as a, np.load(full / "enhanced_train.npz") as b:
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    a = json.loads((only / "enhanced_meta.json").read_text())
    b = json.loads((full / "enhanced_meta.json").read_text())
    for key in ("sequence_audit", "ingestion_audit"):
        assert a[key]["splits"]["train"] == b[key]["splits"]["train"]
    assert "build_scope" not in b
    assert b["asset_audit"]["overlap"] == 0
    assert {p.name for p in full.iterdir()} == {"enhanced_train.npz", "enhanced_val.npz",
                                               "enhanced_meta.json", "replay_display.parquet"}


def test_frozen_contract_required_before_any_archive_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "read_split", lambda *a, **k: pytest.fail("must fail before reading input"))
    with pytest.raises(SystemExit) as error:
        builder.main(["--base-dir", str(tmp_path), "--training-only"])
    assert error.value.code == 2


def test_default_mode_still_rejects_official_robot_overlap(inputs, tmp_path):
    base, metadata = inputs
    write_split(base, "Validation", frames("train-a"))
    output = tmp_path / "full"
    with pytest.raises(ValueError, match="official asset overlap"):
        builder.main(["--base-dir", str(base), "--encoding-meta", str(metadata), "--out-dir", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("field", ["deviceType", "mainState", "errorCode"])
def test_training_unknown_categories_still_fail_before_output(inputs, tmp_path, field):
    base, metadata = inputs
    rows = frames("train-a")
    container = rows[0]["deviceData"] if field == "mainState" else rows[0]["errorData"] if field == "errorCode" else rows[0]
    container[field] = "UNREGISTERED"
    write_split(base, "Training", rows)
    output = tmp_path / "unknown"
    with pytest.raises(ValueError, match=f"unknown {field}"):
        builder.main(arguments(base, metadata, output))
    assert not output.exists()


@pytest.mark.parametrize("issue", ["duplicate", "mixed_clock", "missing_time", "malformed", "no_windows"])
def test_existing_training_guards_still_stop_without_output(inputs, tmp_path, issue):
    base, metadata = inputs
    rows = frames("train-a")
    if issue == "duplicate":
        rows[10]["createdAt"] = rows[9]["createdAt"]
    elif issue == "mixed_clock":
        rows[10]["createdAt"] = "2026-01-01T00:00:10"
    elif issue == "missing_time":
        rows[10]["createdAt"] = None
    elif issue == "malformed":
        rows[10] = b"{"
    else:
        rows = rows[:30]
    write_split(base, "Training", rows)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        builder.main(arguments(base, metadata, output, "--reject-malformed-records"))
    assert not output.exists()


@pytest.mark.parametrize("kind", ["existing", "symlink"])
def test_output_protection_happens_before_input_reads(inputs, tmp_path, monkeypatch, kind):
    base, metadata = inputs
    output = tmp_path / "protected"
    if kind == "existing":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent")
    monkeypatch.setattr(builder, "read_split", lambda *a, **k: pytest.fail("must preserve output before input"))
    with pytest.raises(SystemExit):
        builder.main(arguments(base, metadata, output))
    assert not (tmp_path / "absent").exists()


def test_new_build_runs_synthetic_selection_only_but_not_official_comparison(inputs, tmp_path):
    base, metadata = inputs
    rows = [row for asset in range(6) for row in frames(f"train-{asset}", count=48, all_classes=True)]
    write_split(base, "Training", rows)
    data, champion = tmp_path / "data", tmp_path / "champion"
    builder.main(arguments(base, metadata, data, "--max-gap-seconds", "1"))
    champion.mkdir()
    champion_metadata = json.loads(metadata.read_text())
    (champion / "enhanced_meta.json").write_bytes(metadata.read_bytes())
    names = [name for name, _ in sorted(champion_metadata["err_map"].items(), key=lambda pair: pair[1])]
    (champion / "robot_pdm_enhanced_meta.json").write_text(json.dumps({"class_names": names, "n_features": 249}))
    inputs_before = {p: p.read_bytes() for folder in (data, champion) for p in folder.iterdir()}
    result = trainer.train_candidates(data, tmp_path / "selection", champion, rounds=2,
        normal_ratios=(None,), weight_modes=("unweighted",), selection_only=True, data_kind="synthetic_fixture")
    assert result["selection_only"] and not result["official_validation_evaluated"]
    assert result["evaluation_scope"] == "training_robot_holdout_only"
    assert not result["promotion"]["eligible"] and not result["promotion"]["automatic_promotion"]
    assert set(result["lineage"]["sha256"]) == {"enhanced_train.npz", "enhanced_meta.json"}
    with pytest.raises(FileNotFoundError):
        trainer.train_candidates(data, tmp_path / "official", champion, rounds=1)
    assert not (tmp_path / "official").exists()
    assert {p: p.read_bytes() for p in inputs_before} == inputs_before


def test_failed_npz_write_does_not_publish_metadata(inputs, tmp_path, monkeypatch):
    base, metadata = inputs
    output = tmp_path / "partial"

    def failed_write(stream, **kwargs):
        stream.write(b"partial NPZ")
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez_compressed", failed_write)
    with pytest.raises(OSError, match="simulated write failure"):
        builder.main(arguments(base, metadata, output))
    assert (output / "enhanced_train.npz").read_bytes() == b"partial NPZ"
    assert not (output / "enhanced_meta.json").exists()
    with pytest.raises(SystemExit):
        builder.main(arguments(base, metadata, output))
