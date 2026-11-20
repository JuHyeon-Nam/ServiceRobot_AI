"""Synthetic source auditing tests, not AI-Hub performance evaluations."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import audit_diagnosis_source as audit
import build_enhanced_dataset as builder


def record(asset="a", stamp="2026-01-01T00:00:00Z", label="정상"):
    return {"deviceId": asset, "deviceType": "robot", "createdAt": stamp,
            "deviceData": {"mainState": "MOVE", "batteryLevel": 90, "collision": False,
                           "obstacle": False, "isOffline": False, "nowCharging": False,
                           "emergencyStop": False,
                           "standardLocationData": dict.fromkeys(("x", "y", "degree", "speed"), .2),
                           "totalOperationData": dict.fromkeys(("batteryUse", "batteryCycleCount", "distance"), 1)},
            "siteData": {"crowd": "LOW"}, "errorData": {"errorCode": label}}


def sources(root, train=None, val=None, extras=()):
    for split, rows in (("Training", train or [record()]), ("Validation", val or [record("b")])):
        folder = root / split / "02.라벨링데이터"
        folder.mkdir(parents=True)
        with zipfile.ZipFile(folder / "source.zip", "w") as archive:
            for i, row in enumerate(rows):
                archive.writestr(f"{i}.json", row if isinstance(row, bytes) else json.dumps(row))
            for name, payload in extras:
                archive.writestr(name, payload)
    return root


def frozen_meta(root):
    value = {"dyn": builder.DYN, "stat": builder.STAT_NUM + ["deviceType", "mainState"],
             "crowd_map": builder.CROWD, "err_map": {"정상": 0, "fault": 1},
             "devtype_map": {"robot": 0}, "mainstate_map": {"MOVE": 0}}
    path = root / "meta.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_counts_crc_lineage_and_inputs_preserved_without_building(tmp_path, monkeypatch):
    base = sources(tmp_path / "source", [record("a"), record("c", label="fault")],
                   [record("b", label="fault")], extras=[("readme.txt", b"note"), ("folder/", b"")])
    before = {p: p.read_bytes() for p in base.rglob("*.zip")}
    meta = frozen_meta(tmp_path)
    monkeypatch.setattr(builder, "read_split", lambda *a, **k: pytest.fail("must not retain all sensor records"))
    monkeypatch.setattr(builder, "main", lambda *a, **k: pytest.fail("must not build windows"))
    result = audit.audit_source(base, meta)
    assert not result["current_build_blockers"] and not result["windows_created"]
    assert not result["model_performance_evaluated"] and result["training_readiness"] == "not_evaluated"
    assert result["official_asset_overlap"] == {"count": 0, "assets": []}
    assert result["shared_asset_timestamp_groups"] == 0
    train = result["splits"]["train"]
    assert train["per_class"] == {"fault": {"records": 1, "assets": 1}, "정상": {"records": 1, "assets": 1}}
    assert train["counts"]["members"] == 4 and train["counts"]["non_json_members"] == 2
    assert train["counts"]["timestamped_records"] == 2
    assert not train["numeric_input_issues"] and not any(train["skipped_json"].values())
    assert train["archives"][0]["crc_checked_all_members"]
    path = base / "Training/02.라벨링데이터/source.zip"
    assert train["archives"][0]["sha256"] == hashlib.sha256(before[path]).hexdigest()
    assert result["encoding_audit"]["source_sha256"] == hashlib.sha256(meta.read_bytes()).hexdigest()
    assert {p: p.read_bytes() for p in before} == before
    assert not list(base.rglob("*.json")) and result["resources"]["process_peak_rss_bytes"] > 0
    json.dumps(result, allow_nan=False)


def test_overlap_uses_actual_ids_and_normalized_clocks_not_archive_names(tmp_path):
    base = sources(tmp_path, [record("a"), record("a", "2026-01-01T09:00:00+09:00", "fault"),
                              record("a", "2026-01-01T00:02:00Z")], [record("a"), record("a")])
    result = audit.audit_source(base)
    assert result["official_asset_overlap"] == {"count": 1, "assets": ["a"]}
    assert result["shared_asset_timestamp_groups"] == 1
    clock = result["splits"]["train"]["clock"]
    assert clock["duplicate_timestamp_groups"] == 1 and clock["duplicate_extra_records"] == 1
    assert clock["conflicting_label_timestamp_groups"] == 1
    assert clock["max_gap_seconds"] == 120
    assert clock["sorted_adjacent_gap_counts"] == {"zero": 1, "over_60s": 1}
    assert {i["code"] for i in result["current_build_blockers"]} == {"official_asset_overlap", "duplicate_timestamp_groups"}


def test_order_mixed_clocks_and_invalid_id_timestamp_are_distinct(tmp_path):
    rows = [record("a", "2026-01-01T00:00:02Z"), record("a", "2026-01-01T00:00:01Z"),
            record("a", "2026-01-01T00:00:00"), record("a", "bad"), record(" a ")]
    result = audit.audit_source(sources(tmp_path, rows))["splits"]["train"]
    assert result["counts"]["out_of_order_arrival_steps"] == 1
    assert result["counts"]["invalid_asset_records"] == 1
    assert result["counts"]["invalid_timestamp_records"] == 1
    assert result["counts"]["timestamped_records"] == 3
    assert result["clock"]["mixed_clock_assets"] == ["a"]
    assert result["clock"]["sorted_adjacent_gap_counts"] == {"up_to_1s": 1}


@pytest.mark.parametrize("value", [None, 4, "", "bad", "NaT"])
def test_invalid_timestamps_not_coerced_to_valid_times(tmp_path, value):
    result = audit.audit_source(sources(tmp_path, [record(stamp=value)]))
    assert result["splits"]["train"]["counts"]["invalid_timestamp_records"] == 1
    assert result["splits"]["train"]["counts"]["timestamped_records"] == 0


def test_all_known_ingestion_exclusions_match_builder(tmp_path):
    base = sources(tmp_path, [record(), b"\xff", b"{", [], record(label=None), record(label="NOT_ASSIGNED")])
    result = audit.audit_source(base)["splits"]["train"]
    assert result["counts"]["json_members"] == 6 and result["counts"]["label_kept_records"] == 1
    assert result["skipped_json"] == dict.fromkeys(builder.INGEST_SKIP_REASONS, 1)


def test_numeric_defaults_unknown_categories_and_missing_class_support_are_not_hidden(tmp_path):
    row = record()
    row["deviceData"]["batteryLevel"] = "invalid"
    row["deviceData"]["standardLocationData"]["speed"] = float("nan")
    row["deviceData"]["totalOperationData"]["distance"] = 1e100
    row["siteData"]["crowd"] = "other"
    row["deviceType"] = "new_type"
    row["deviceData"]["mainState"] = None
    result = audit.audit_source(sources(tmp_path / "source", [row]), frozen_meta(tmp_path))
    train = result["splits"]["train"]
    assert train["numeric_input_issues"] == {"batteryLevel:missing_or_invalid_defaults_to_zero": 1,
        "speed:nonfinite_or_outside_float32": 1, "distance:nonfinite_or_outside_float32": 1}
    assert train["counts"]["crowd_defaults_to_middle"] == 1
    assert train["invalid_categories"] == {"mainState": 1}
    assert train["unknown_frozen_categories"] == {"deviceType": {"new_type": 1}}
    assert train["per_class"]["fault"] == {"records": 0, "assets": 0}
    json.dumps(result, allow_nan=False)


def test_duplicate_member_names_are_read_by_entry(tmp_path):
    base = sources(tmp_path)
    path = base / "Training/02.라벨링데이터/source.zip"
    with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("0.json", json.dumps(record("c", label="fault")))
    train = audit.audit_source(base)["splits"]["train"]
    assert train["per_class"] == {"fault": {"records": 1, "assets": 1}, "정상": {"records": 1, "assets": 1}}


@pytest.mark.parametrize("non_json", [False, True])
def test_crc_failure_including_non_json_aborts_without_report(tmp_path, non_json):
    base = sources(tmp_path / "source", extras=[("note.txt", b"unique-note")])
    path = base / "Training/02.라벨링데이터/source.zip"
    content = path.read_bytes()
    old = b"unique-note" if non_json else json.dumps(record()).encode()
    assert content.count(old) == 1
    path.write_bytes(content.replace(old, b"!" + old[1:]))
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as error:
        audit.main(["--base-dir", str(base), "--report", str(output)])
    assert error.value.code == 2 and not output.exists()


def test_json_budget_checked_before_payload_read(tmp_path, monkeypatch):
    base = sources(tmp_path)
    monkeypatch.setattr(audit, "MAX_JSON_BYTES", 1)
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda *a, **k: pytest.fail("must check declared size first"))
    with pytest.raises(ValueError, match="byte budget"):
        audit.audit_source(base)


def test_cli_saves_blocked_audit_but_does_not_train_or_overwrite(tmp_path, monkeypatch):
    base = sources(tmp_path / "source", val=[record()])
    output = tmp_path / "reports/result.json"
    assert audit.main(["--base-dir", str(base), "--report", str(output)]) == 2
    before = output.read_bytes()
    assert json.loads(before)["official_asset_overlap"]["count"] == 1
    monkeypatch.setattr(audit, "audit_source", lambda *a, **k: pytest.fail("check output before input"))
    with pytest.raises(SystemExit) as error:
        audit.main(["--base-dir", str(base), "--report", str(output)])
    assert error.value.code == 2 and output.read_bytes() == before


def test_cli_success_and_dangling_symlink_protection(tmp_path):
    base = sources(tmp_path / "source")
    output = tmp_path / "result.json"
    assert audit.main(["--base-dir", str(base), "--report", str(output)]) == 0
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(SystemExit):
        audit.main(["--base-dir", str(base), "--report", str(link)])
    assert not (tmp_path / "absent").exists()


def test_source_mutation_is_rejected(tmp_path, monkeypatch):
    base = sources(tmp_path)
    original = audit._sha256

    def mutate(path):
        fingerprint = original(path)
        with path.open("ab") as stream:
            stream.write(b"changed after hash")
        return fingerprint

    monkeypatch.setattr(audit, "_sha256", mutate)
    with pytest.raises(ValueError, match="source changed"):
        audit.audit_source(base)


def test_missing_split_is_not_reported_as_zero_overlap(tmp_path):
    with pytest.raises(FileNotFoundError, match="no label JSON"):
        audit.audit_source(tmp_path)


def test_unexpected_parser_failure_is_not_a_malformed_skip(tmp_path, monkeypatch):
    base = sources(tmp_path)
    monkeypatch.setattr(audit, "parse_record", lambda _: (_ for _ in ()).throw(RuntimeError("unexpected")))
    with pytest.raises(RuntimeError, match="unexpected"):
        audit.audit_source(base)
