"""Ingestion accounting on synthetic ZIPs; not evidence of AI-Hub model performance."""
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


def record(asset="robot-a", index=0, code="정상"):
    return {"deviceId": asset, "deviceType": "robot",
            "createdAt": f"2026-01-01T00:00:{index:02d}Z", "errorData": {"errorCode": code},
            "deviceData": {"mainState": "MOVE", "batteryLevel": 90 - index,
                           "standardLocationData": {"x": index, "y": index / 10, "speed": .2}}}


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content if isinstance(content, bytes) else json.dumps(content))


def assert_accounting(counts):
    assert counts["members"] == counts["json_members"] + counts["non_json_members"]
    assert counts["json_members"] == counts["accepted_records"] + sum(counts["skipped_json"].values())
    assert counts["accepted_records"] == sum(counts["accepted_by_class"].values())
    assert set(counts["skipped_json"]) == set(builder.INGEST_SKIP_REASONS)


def test_per_archive_and_split_counts_partition_every_member(tmp_path, capsys):
    write_zip(tmp_path / "b.zip", [("normal.json", record(index=2)),
                                   ("fault.json", record(index=3, code="E-RBT-B")),
                                   ("not-assigned.json", record(code="NOT_ASSIGNED")),
                                   ("image.png", b"not a JSON record")])
    write_zip(tmp_path / "a.zip", [("normal.json", record(index=0)),
                                   ("missing-label.json", record(code=None)),
                                   ("broken.json", b"{"), ("encoding.json", b"\xff"),
                                   ("structure.json", []), ("ignored.JSON", record()),
                                   ("directory/", b"")])
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    audit = {}
    by_asset, types, states = builder.read_split(tmp_path, "train", audit=audit)
    assert types == {"robot"} and states == {"MOVE"}
    assert [row["batteryLevel"] for row in by_asset["robot-a"]] == [90, 88, 87]
    assert audit["archive_count"] == 2
    assert [entry["archive"] for entry in audit["archives"]] == ["a.zip", "b.zip"]
    assert {key: audit[key] for key in ("members", "json_members", "non_json_members", "accepted_records")} == {
        "members": 11, "json_members": 8, "non_json_members": 3, "accepted_records": 3}
    assert audit["accepted_by_class"] == {"정상": 2, "E-RBT-B": 1}
    assert audit["skipped_json"] == dict.fromkeys(builder.INGEST_SKIP_REASONS, 1)
    assert_accounting(audit)
    for entry in audit["archives"]:
        assert_accounting(entry)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    output = capsys.readouterr().out
    assert "제외 4건" in output and "제외 1건" in output
    assert "invalid_utf8" in output and "not_assigned_error_code" in output


@pytest.mark.parametrize("payload,reason", [
    (b"\xff", "invalid_utf8"), (b"{", "invalid_json"), (b"", "invalid_json"),
    (None, "record_parse_error"), ([], "record_parse_error"), (1, "record_parse_error"),
    ({"deviceData": None}, "record_parse_error"),
    ({"deviceData": {"standardLocationData": []}}, "record_parse_error"),
    ({"deviceData": {"totalOperationData": None}}, "record_parse_error"),
    ({"siteData": None}, "record_parse_error"),
    ({"siteData": {"crowd": []}}, "record_parse_error"),
    ({"errorData": []}, "record_parse_error"),
    ({}, "missing_error_code"), (record(code=None), "missing_error_code"),
    (record(code="NOT_ASSIGNED"), "not_assigned_error_code"),
])
def test_known_exclusions_are_counted_once_without_fabricating_records(tmp_path, payload, reason):
    write_zip(tmp_path / "records.zip", [("excluded.json", payload), ("accepted.json", record())])
    audit = {}
    by_asset, _, _ = builder.read_split(tmp_path, "train", audit=audit)
    assert by_asset["robot-a"] == [builder.parse_record(record())]
    assert audit["accepted_records"] == 1
    assert audit["accepted_by_class"] == {"정상": 1}
    assert audit["skipped_json"] == {key: int(key == reason) for key in builder.INGEST_SKIP_REASONS}
    assert_accounting(audit)


def test_default_return_contract_and_numeric_fallbacks_are_unchanged(tmp_path):
    raw = record()
    raw["deviceData"]["batteryLevel"] = "not-a-number"
    raw["siteData"] = {"crowd": "unrecognized"}
    write_zip(tmp_path / "records.zip", [("record.json", raw)])
    legacy_result = builder.read_split(tmp_path, "train")
    audit = {}
    assert builder.read_split(tmp_path, "train", audit=audit) == legacy_result
    parsed = legacy_result[0]["robot-a"][0]
    assert parsed["batteryLevel"] == 0.0
    assert parsed["crowd"] == 1.0
    assert audit["accepted_records"] == 1
    assert sum(audit["skipped_json"].values()) == 0


def test_duplicate_filenames_read_their_own_zip_entries_not_the_last_entry_twice(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        write_zip(tmp_path / "records.zip", [("same.json", record(index=0)),
                                             ("same.json", record(index=1, code="E-RBT-B"))])
    audit = {}
    by_asset, _, _ = builder.read_split(tmp_path, "train", audit=audit)
    assert [row["batteryLevel"] for row in by_asset["robot-a"]] == [90, 89]
    assert [row["errorCode"] for row in by_asset["robot-a"]] == ["정상", "E-RBT-B"]
    assert audit["accepted_by_class"] == {"정상": 1, "E-RBT-B": 1}
    assert_accounting(audit)


@pytest.mark.parametrize("error_type", [OSError, zipfile.BadZipFile, RuntimeError, MemoryError])
def test_archive_read_errors_abort_instead_of_becoming_record_skips(tmp_path, monkeypatch, error_type):
    write_zip(tmp_path / "records.zip", [("first.json", record()), ("broken.json", record(index=1))])
    original_read = builder.zipfile.ZipFile.read
    audit = {"prior_result": "unchanged"}

    def interrupted_read(archive, member, *args, **kwargs):
        if member.filename == "broken.json":
            raise error_type("simulated archive read failure")
        return original_read(archive, member, *args, **kwargs)

    monkeypatch.setattr(builder.zipfile.ZipFile, "read", interrupted_read)
    with pytest.raises(error_type, match="simulated archive read failure"):
        builder.read_split(tmp_path, "train", audit=audit)
    assert audit == {"prior_result": "unchanged"}


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, MemoryError])
def test_unexpected_parser_errors_are_not_silently_skipped(tmp_path, monkeypatch, error_type):
    write_zip(tmp_path / "records.zip", [("record.json", record())])
    audit = {}

    def unexpected_error(document):
        raise error_type("simulated parser failure")

    monkeypatch.setattr(builder, "parse_record", unexpected_error)
    with pytest.raises(error_type, match="simulated parser failure"):
        builder.read_split(tmp_path, "train", audit=audit)
    assert not audit


def test_corrupt_archive_stops_cli_before_output_creation(tmp_path, capsys):
    folder = tmp_path / "source" / "Training" / "02.라벨링데이터"
    folder.mkdir(parents=True)
    corrupt = folder / "corrupt.zip"
    corrupt.write_bytes(b"not a zip archive")
    output = tmp_path / "output"
    with pytest.raises(zipfile.BadZipFile):
        builder.main(["--base-dir", str(tmp_path / "source"), "--out-dir", str(output)])
    assert not output.exists()
    assert corrupt.read_bytes() == b"not a zip archive"
    assert "저장 완료" not in capsys.readouterr().out


def test_real_zip_crc_failure_is_not_silently_treated_as_bad_json(tmp_path):
    folder = tmp_path / "source" / "Training" / "02.라벨링데이터"
    folder.mkdir(parents=True)
    path = folder / "records.zip"
    document = record()
    write_zip(path, [("record.json", document)])
    payload = json.dumps(document).encode("utf-8")
    original = path.read_bytes()
    assert original.count(payload) == 1
    corrupted = original.replace(payload, b"!" + payload[1:], 1)
    path.write_bytes(corrupted)
    output = tmp_path / "output"
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        builder.main(["--base-dir", str(tmp_path / "source"), "--out-dir", str(output)])
    assert not output.exists()
    assert path.read_bytes() == corrupted


def test_build_metadata_links_raw_ingestion_to_window_counts_and_preserves_values(tmp_path):
    source = tmp_path / "source"
    source_before = {}
    for split, asset in (("Training", "train-robot"), ("Validation", "val-robot")):
        folder = source / split / "02.라벨링데이터"
        folder.mkdir(parents=True)
        accepted = [(f"{i}.json", record(asset, i, "E-RBT-B" if i % 2 else "정상"))
                    for i in range(35)]
        excluded = [("bad-encoding.json", b"\xff"), ("bad-json.json", b"{"),
                    ("bad-record.json", []), ("missing-label.json", record(code=None)),
                    ("unassigned.json", record(code="NOT_ASSIGNED")), ("notes.txt", b"notes")]
        path = folder / "records.zip"
        write_zip(path, [*reversed(accepted), *excluded])
        source_before[path] = path.read_bytes()
    output = tmp_path / "output"
    builder.main(["--base-dir", str(source), "--out-dir", str(output)])
    metadata = json.loads((output / "enhanced_meta.json").read_text())
    ingestion = metadata["ingestion_audit"]
    assert ingestion["schema"] == "servicerobot.dataset.ingestion.v1"
    assert ingestion["json_member_policy"] == "case_sensitive_.json_suffix; each_zip_entry"
    for split, filename, asset in (("train", "train", "train-robot"), ("validation", "val", "val-robot")):
        counts = ingestion["splits"][split]
        assert_accounting(counts)
        assert counts["json_members"] == 40 and counts["non_json_members"] == 1
        assert counts["accepted_records"] == metadata["sequence_audit"]["splits"][split]["records"] == 35
        assert counts["accepted_by_class"] == {"정상": 18, "E-RBT-B": 17}
        assert counts["skipped_json"] == dict.fromkeys(builder.INGEST_SKIP_REASONS, 1)
        loaded = load_dataset(output / f"enhanced_{filename}.npz")
        assert loaded["features"].shape == (5, 249)
        assert loaded["groups"].tolist() == [asset] * 5
        expected_codes = ["E-RBT-B" if i % 2 else "정상" for i in range(30, 35)]
        assert loaded["y"].tolist() == [metadata["err_map"][code] for code in expected_codes]
        np.testing.assert_array_equal(loaded["X"][:, -1, 0], np.arange(61, 56, -1))
        assert metadata["sequence_audit"]["splits"][split]["emitted_windows"] == 5
    display = pd.read_parquet(output / "replay_display.parquet")
    assert display.validx.tolist() == list(range(5))
    assert display.x.tolist() == list(range(30, 35))
    assert {path: path.read_bytes() for path in source_before} == source_before


def test_all_excluded_records_produce_honest_zero_counts_not_normal_samples(tmp_path):
    write_zip(tmp_path / "records.zip", [("bad.json", b"{"), ("unassigned.json", record(code="NOT_ASSIGNED"))])
    audit = {}
    by_asset, types, states = builder.read_split(tmp_path, "train", audit=audit)
    assert not by_asset and not types and not states
    assert audit["accepted_records"] == 0
    assert audit["accepted_by_class"] == {}
    assert sum(audit["skipped_json"].values()) == 2
    assert_accounting(audit)
