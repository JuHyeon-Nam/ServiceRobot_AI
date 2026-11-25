"""Synthetic pre-execution proposals; no protocol approval or model-performance evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import audit_diagnosis_temporal as inspector
import train_diagnosis as trainer


@pytest.fixture
def inputs(tmp_path):
    data, champion = tmp_path / "data", tmp_path / "champion"
    data.mkdir()
    champion.mkdir()
    meta = json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())
    (champion / "enhanced_meta.json").write_text(json.dumps(meta))
    names = [name for name, _ in sorted(meta["err_map"].items(), key=lambda item: item[1])]
    (champion / "robot_pdm_enhanced_meta.json").write_text(json.dumps({"class_names": names, "n_features": 249}))
    meta.update({"window_contract": {"history": "30 observations before target", "context": "at target time"},
        "sequence_audit": {"max_gap_seconds": 10, "duplicate_policy": "reject",
                          "timestamp_policy": "consistent_asset_clock; aware_times_normalized_to_utc"},
        "build_scope": {"mode": "training_only", "official_validation_read": False,
                        "official_validation_windows_created": False}})
    (data / "enhanced_meta.json").write_text(json.dumps(meta))
    y = np.tile(np.repeat(np.arange(len(names)), 2), 12)
    groups = np.repeat([f"robot-{i}" for i in range(6)], len(names) * 4)
    seconds = np.tile(np.r_[np.arange(18), 600 + np.arange(18)], 6)
    times = np.asarray([(pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(seconds=int(s))).isoformat()
                        for s in seconds])
    np.savez_compressed(data / "enhanced_train.npz", X=np.zeros((len(y), 30, 7), dtype=np.float32),
        S=np.zeros((len(y), 9), dtype=np.float32), y=y, groups=groups, target_times=times)
    return data, champion


def audit(inputs, **kwargs):
    return inspector.audit_temporal(*inputs, cutoff="2026-01-01T00:05:00Z", **kwargs)


def rewrite(data, **updates):
    path = data / "enhanced_train.npz"
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    arrays.update(updates)
    np.savez_compressed(path, **arrays)


def test_only_descriptors_loaded_no_model_features_or_official_validation_and_inputs_preserved(inputs, monkeypatch):
    data, champion = inputs
    (data / "enhanced_val.npz").write_bytes(b"invalid official Validation")
    (champion / "robot_pdm_enhanced.txt").write_bytes(b"not a Booster")
    before = {p: p.read_bytes() for folder in inputs for p in folder.iterdir()}
    loaded, original = [], np.lib.npyio.NpzFile.__getitem__

    def tracked(archive, key):
        assert key in {"y", "groups", "target_times"}
        loaded.append(key)
        return original(archive, key)

    def forbidden(*args, **kwargs):
        pytest.fail("preparation must not load sensors, compute features, fit, or load a Booster")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracked)
    monkeypatch.setattr(trainer, "make_feature_matrix_batched", forbidden)
    monkeypatch.setattr(trainer, "load_dataset", forbidden)
    monkeypatch.setattr(trainer, "load_booster", forbidden)
    monkeypatch.setattr(trainer.lgb.LGBMClassifier, "fit", forbidden)
    report = audit(inputs, min_selection_rows=2)
    assert loaded == ["y", "groups", "target_times"]
    assert report["preparation_checks_passed"] and report["requires_user_scope_approval"]
    assert not report["authorized_for_training"] and report["protocol_status"] == "proposal_not_approved_or_selected"
    assert report["plan"]["fit_rows"] == report["plan"]["selection_rows"] == 108
    assert not report["official_validation_read"] and not report["sensor_array_payloads_loaded"]
    assert not report["model_trained"] and not report["feature_matrix_created"]
    assert report["official_validation_metrics"] is None and not report["full_npz_crc_checked"]
    assert {p: p.read_bytes() for p in before} == before


def test_support_failure_never_changes_cutoff_or_searches_for_more_favorable_labels(inputs, monkeypatch):
    original, calls = inspector.purged_temporal_holdout, []

    def tracked(*args, **kwargs):
        calls.append(kwargs["cutoff"])
        return original(*args, **kwargs)

    monkeypatch.setattr(inspector, "purged_temporal_holdout", tracked)
    report = audit(inputs)
    assert not report["preparation_checks_passed"] and len(report["selection_support"]["issues"]) == 9
    rewrite(inputs[0], y=np.zeros(216, dtype=np.int64))
    other = audit(inputs)
    assert report["plan"] == other["plan"]
    assert calls == ["2026-01-01T00:05:00Z"] * 2
    assert not other["fit_support"]["sufficient_support"] and not other["preparation_checks_passed"]


@pytest.mark.parametrize("issue", ["not_training", "no_gap", "bad_metadata", "unknown_label", "numeric_time", "duplicate_time"])
def test_incompatible_contract_or_descriptors_rejected(inputs, issue):
    data, _ = inputs
    meta_path = data / "enhanced_meta.json"
    meta = json.loads(meta_path.read_text())
    if issue == "not_training":
        meta["build_scope"]["official_validation_read"] = True
    elif issue == "no_gap":
        meta["sequence_audit"]["max_gap_seconds"] = None
    elif issue == "bad_metadata":
        meta = []
    elif issue == "unknown_label":
        rewrite(data, y=np.full(216, 99, dtype=np.int64))
    elif issue == "numeric_time":
        rewrite(data, target_times=np.arange(216))
    else:
        rewrite(data, target_times=np.repeat("2026-01-01T00:00:00Z", 216))
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        audit(inputs)


def test_descriptor_budget_is_checked_before_numpy_payload_loading(inputs, monkeypatch):
    monkeypatch.setattr(inspector, "MAX_DESCRIPTOR_BYTES", 1)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("budget must be checked first"))
    with pytest.raises(ValueError, match="budget"):
        audit(inputs)


@pytest.mark.parametrize("cutoff,minimum", [("2026-01-01T00:00:00", 20), ("bad", 20), ("2026-01-01T00:00:00Z", 0)])
def test_invalid_policy_or_naive_cutoff_rejected_before_file_io(inputs, monkeypatch, cutoff, minimum):
    monkeypatch.setattr(inspector, "file_sha256", lambda *a: pytest.fail("validate options before input"))
    with pytest.raises(ValueError):
        inspector.audit_temporal(*inputs, cutoff=cutoff, min_selection_rows=minimum)


@pytest.mark.parametrize("minimum,expected", [(2, 0), (20, 2)])
def test_cli_saves_passing_or_insufficient_proposal_without_approval_and_protects_report(inputs, tmp_path, minimum, expected):
    data, champion = inputs
    output = tmp_path / "report.json"
    args = ["--data-dir", str(data), "--champion-dir", str(champion), "--cutoff", "2026-01-01T00:05:00Z",
            "--min-selection-rows", str(minimum), "--report", str(output)]
    assert inspector.main(args) == expected
    saved = output.read_bytes()
    assert not json.loads(saved)["authorized_for_training"]
    with pytest.raises(SystemExit):
        inspector.main(args)
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(SystemExit):
        inspector.main([*args[:-1], str(link)])
    assert output.read_bytes() == saved and not (tmp_path / "absent").exists()


def test_source_change_during_preparation_is_rejected(inputs, monkeypatch):
    original = inspector.purged_temporal_holdout

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with (inputs[0] / "enhanced_train.npz").open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(inspector, "purged_temporal_holdout", mutate)
    with pytest.raises(ValueError, match="changed during"):
        audit(inputs)
