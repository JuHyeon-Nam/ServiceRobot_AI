"""Temporal window contracts on synthetic records/ZIPs, not AI-Hub performance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_enhanced_dataset as builder
from train_diagnosis import load_dataset


def records(times):
    return [{"createdAt": timestamp, "value": i} for i, timestamp in enumerate(times)]


def write_archives(root, times):
    for split, asset in (("Training", "train-robot"), ("Validation", "validation-robot")):
        folder = root / split / "02.라벨링데이터"
        folder.mkdir(parents=True)
        with zipfile.ZipFile(folder / "records.zip", "w") as archive:
            for i in reversed(range(len(times))):
                archive.writestr(f"{i}.json", json.dumps({
                    "deviceId": asset, "deviceType": "robot", "createdAt": times[i],
                    "errorData": {"errorCode": "정상"},
                    "deviceData": {"mainState": "MOVE", "batteryLevel": i,
                                   "standardLocationData": {"x": i, "y": i / 10, "speed": .2}},
                }))


def test_aware_timestamps_are_sorted_by_instant_without_mutating_records():
    rows = records(["2026-01-01T00:00:01Z", "2026-01-01T09:00:00+09:00",
                    "2025-12-31T19:00:02-05:00"])
    before = deepcopy(rows)
    ordered, sessions, audit = builder.prepare_sequence(rows, "robot-a", 1)
    assert [row["value"] for row in ordered] == [1, 0, 2]
    np.testing.assert_array_equal(sessions, [0, 0, 0])
    assert audit == {"timestamp_basis": "utc", "gaps": 0}
    assert rows == before
    assert ordered[0]["createdAt"] == "2026-01-01T09:00:00+09:00"


def test_naive_timestamps_are_not_reported_as_verified_utc():
    ordered, sessions, audit = builder.prepare_sequence(
        records(["2026-01-01T00:00:01", "2026-01-01T00:00:00"]), "robot-a", 1)
    assert [row["value"] for row in ordered] == [1, 0]
    assert audit == {"timestamp_basis": "naive_shared_clock", "gaps": 0}
    np.testing.assert_array_equal(sessions, [0, 0])


@pytest.mark.parametrize("asset", [None, 1, True, "", " ", " robot", "robot ", "NaN", "null", "NaT"])
def test_ambiguous_asset_ids_are_not_stringified(asset):
    with pytest.raises(ValueError, match="deviceId must be"):
        builder.prepare_sequence(records(["2026-01-01T00:00:00Z"]), asset)


@pytest.mark.parametrize("timestamp", [None, "", " ", 123, "not-a-time", "NaT",
                                        "2026-01-01T00:00:60Z", "01/02/2026"])
def test_missing_or_non_iso_timestamps_cannot_form_windows(timestamp):
    with pytest.raises(ValueError, match="createdAt for robot-a"):
        builder.prepare_sequence(records([timestamp]), "robot-a")


@pytest.mark.parametrize("times", [
    ["2026-01-01T00:00:00", "2026-01-01T00:00:00"],
    ["2026-01-01T00:00:00Z", "2026-01-01T09:00:00+09:00"],
])
def test_duplicate_instants_are_rejected_including_different_offsets(times):
    with pytest.raises(ValueError, match="duplicate createdAt"):
        builder.prepare_sequence(records(times), "robot-a")


def test_mixed_timezone_awareness_is_rejected():
    with pytest.raises(ValueError, match="mixed timezone-aware and naive"):
        builder.prepare_sequence(records(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01"]), "robot-a")


def test_gap_boundary_keeps_nanosecond_precision():
    times = ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00.000000001Z",
             "2026-01-01T00:00:00.000000003Z"]
    _, sessions, audit = builder.prepare_sequence(records(times), "robot-a", 1e-9)
    np.testing.assert_array_equal(sessions, [0, 0, 1])
    assert audit["gaps"] == 1


def test_gap_filtering_is_disabled_without_a_declared_threshold():
    _, sessions, audit = builder.prepare_sequence(
        records(["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"]), "robot-a")
    np.testing.assert_array_equal(sessions, [0, 0])
    assert audit["gaps"] == 0


@pytest.mark.parametrize("gap", [0, -1, float("nan"), float("inf"), True, "1", 1e-12, 1e30])
def test_invalid_gap_thresholds_are_rejected(gap):
    with pytest.raises(ValueError, match="max gap seconds"):
        builder.prepare_sequence(records(["2026-01-01T00:00:00Z"]), "robot-a", gap)


@pytest.mark.parametrize("gap", ["0", "-1", "nan", "inf", "1e-12", "1e30"])
def test_cli_rejects_bad_gap_configuration_before_archive_io(tmp_path, monkeypatch, gap):
    def forbidden(*args):
        pytest.fail("invalid gap must be rejected before reading archives")

    monkeypatch.setattr(builder, "read_split", forbidden)
    with pytest.raises(SystemExit) as error:
        builder.main(["--base-dir", str(tmp_path), "--max-gap-seconds", gap])
    assert error.value.code == 2


def test_empty_sequence_has_no_timestamp_basis():
    with pytest.raises(ValueError, match="no timestamped records"):
        builder.prepare_sequence([], "robot-a")


@pytest.mark.parametrize("gap,expected_rows,discarded", [(None, 32, 0), (1, 2, 30)])
def test_zip_windows_audit_gaps_and_keep_replay_target_alignment(tmp_path, gap, expected_rows, discarded):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    moments = [start + pd.Timedelta(seconds=i) for i in [*range(31), *range(120, 151)]]
    times = [(time.tz_convert("Asia/Seoul") if i % 2 == 0 else time).isoformat()
             for i, time in enumerate(moments)]
    write_archives(tmp_path / "source", times)
    output = tmp_path / "built"
    args = ["--base-dir", str(tmp_path / "source"), "--out-dir", str(output)]
    if gap is not None:
        args += ["--max-gap-seconds", str(gap)]
    builder.main(args)
    metadata = json.loads((output / "enhanced_meta.json").read_text())
    audit = metadata["sequence_audit"]
    assert audit["max_gap_seconds"] == gap
    assert audit["duplicate_policy"] == "reject"
    for split, asset in (("train", "train-robot"), ("validation", "validation-robot")):
        entry = audit["splits"][split]
        assert entry == {"records": 62, "input_assets": 1, "gaps": int(gap is not None),
                         "candidate_windows": 32, "discarded_gap_windows": discarded,
                         "emitted_windows": expected_rows, "timestamp_basis_assets": {"utc": 1, "naive_shared_clock": 0}}
        filename = "train" if split == "train" else "val"
        loaded = load_dataset(output / f"enhanced_{filename}.npz")
        assert loaded["features"].shape == (expected_rows, 249)
        assert set(loaded["groups"]) == {asset}
    display = pd.read_parquet(output / "replay_display.parquet")
    assert display.validx.tolist() == list(range(expected_rows))
    assert display.robot.tolist() == ["validation-robot"] * expected_rows
    expected_targets = [30, 61] if gap is not None else list(range(30, 62))
    assert display.x.tolist() == expected_targets
    with np.load(output / "enhanced_val.npz") as arrays:
        assert arrays["target_times"].tolist() == [times[i] for i in expected_targets]
        np.testing.assert_array_equal(arrays["X"][:, -1, 0], np.array(expected_targets) - 1)
        if gap is not None:
            np.testing.assert_array_equal(arrays["X"][:, 0, 0], [0, 31])


@pytest.mark.parametrize("kind,message", [("duplicate", "duplicate createdAt"), ("invalid", "invalid ISO8601"),
                                           ("missing", "missing createdAt"), ("mixed", "mixed timezone-aware")])
def test_bad_archive_times_fail_before_any_dataset_is_written(tmp_path, kind, message):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    times = [(start + pd.Timedelta(seconds=i)).isoformat() for i in range(35)]
    times[10] = {"duplicate": times[9], "invalid": "not-a-time", "missing": None,
                 "mixed": "2026-01-01T00:00:10"}[kind]
    write_archives(tmp_path / "source", times)
    output = tmp_path / "invalid-output"
    with pytest.raises(ValueError, match=message):
        builder.main(["--base-dir", str(tmp_path / "source"), "--out-dir", str(output)])
    assert not output.exists()


def test_all_short_sessions_fail_without_emitting_cross_gap_windows(tmp_path):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    times = [(start + pd.Timedelta(seconds=i)).isoformat() for i in [*range(30), *range(120, 150)]]
    write_archives(tmp_path / "source", times)
    output = tmp_path / "too-short"
    with pytest.raises(ValueError, match="both splits must contain sensor windows"):
        builder.main(["--base-dir", str(tmp_path / "source"), "--out-dir", str(output), "--max-gap-seconds", "1"])
    assert not output.exists()


def test_last_update_time_fallback_is_preserved():
    rows = [builder.parse_record({"lastUpdateTime": f"2026-01-01T00:00:0{i}Z"}) for i in (1, 0)]
    ordered, _, audit = builder.prepare_sequence(rows, "robot-a")
    assert ordered[0]["createdAt"].endswith("00Z")
    assert audit["timestamp_basis"] == "utc"
