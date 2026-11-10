"""Data readiness checks use synthetic arrays, never real diagnosis-performance evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_diagnosis import audit_training_data, load_dataset, main, train_candidates

SERVING = ROOT / "data/processed"


@pytest.fixture
def robot_data(tmp_path):
    data = tmp_path / "dataset"
    data.mkdir()
    metadata = json.loads((SERVING / "enhanced_meta.json").read_text(encoding="utf-8"))
    (data / "enhanced_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    per_asset = np.repeat(np.arange(len(metadata["err_map"])), 2)
    for split in ("train", "val"):
        labels = np.tile(per_asset, 6)
        groups = np.repeat([f"{split}-{i}" for i in range(6)], len(per_asset))
        np.savez(data / f"enhanced_{split}.npz", X=np.zeros((len(labels), 30, 7), dtype=np.float32),
                 S=np.zeros((len(labels), 9), dtype=np.float32), y=labels, groups=groups)
    return data


def replace_arrays(path, **changes):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays.update(changes)
    np.savez(path, **arrays)


def test_audit_reports_ready_without_features_training_or_model_mutation(robot_data, monkeypatch):
    import train_diagnosis

    def unexpected(*args, **kwargs):
        pytest.fail("preflight must not train, infer, or build the full feature matrix")

    before = (SERVING / "robot_pdm_enhanced.txt").read_bytes()
    original_load = train_diagnosis.load_dataset
    loaded_splits = []

    def track_load(path, **kwargs):
        assert kwargs == {"with_features": False, "keep_raw": False}
        rows = original_load(path, **kwargs)
        assert set(rows) == {"y", "groups"}
        loaded_splits.append(path.name)
        return rows

    monkeypatch.setattr(train_diagnosis, "load_dataset", track_load)
    monkeypatch.setattr(train_diagnosis, "make_feature_matrix_batched", unexpected)
    monkeypatch.setattr(train_diagnosis, "train_candidates", unexpected)
    monkeypatch.setattr(train_diagnosis, "load_booster", unexpected)
    report = audit_training_data(robot_data, SERVING, data_kind="synthetic_fixture")
    assert loaded_splits == ["enhanced_train.npz", "enhanced_val.npz"]
    assert not report["raw_sensor_arrays_retained"]
    assert report["ready_for_candidate_comparison"]
    assert not report["model_performance_evaluated"]
    assert report["issues"] == []
    assert report["splits"]["train"]["rows"] == 108
    assert report["splits"]["train"]["per_class"]["E-RBT-S"] == {"rows": 12, "assets": 6}
    plan = report["selection_plan"]
    assert not set(plan["fit_assets"]) & set(plan["selection_assets"])
    assert not plan["official_validation_used_for_selection"]
    assert all(row["rows"] > 0 for row in plan["selection_support"].values())
    assert before == (SERVING / "robot_pdm_enhanced.txt").read_bytes()


def test_audit_lists_all_missing_inputs(tmp_path):
    report = audit_training_data(tmp_path / "missing-data", tmp_path / "missing-model")
    assert not report["ready_for_candidate_comparison"]
    assert len(report["issues"]) == 6
    assert all(issue["code"] == "missing_file" for issue in report["issues"])


@pytest.mark.parametrize("keep_raw", [True, False])
def test_numeric_validation_checks_values_after_the_first_batch(tmp_path, keep_raw):
    raw = np.zeros((4097, 30, 7), dtype=np.float32)
    raw[-1, -1, -1] = np.nan
    path = tmp_path / "last-batch.npz"
    np.savez(path, X=raw, S=np.zeros((4097, 9)), y=np.zeros(4097, dtype=int),
             groups=np.full(4097, "robot"))
    with pytest.raises(ValueError, match="finite"):
        load_dataset(path, with_features=False, keep_raw=keep_raw)


@pytest.mark.parametrize("bad_group", ["", " ", "nan", "NaN", "None", "null", "NaT", " robot", "robot "])
def test_stringified_missing_or_ambiguous_asset_ids_are_rejected(robot_data, bad_group):
    path = robot_data / "enhanced_train.npz"
    replace_arrays(path, groups=np.full(108, bad_group))
    with pytest.raises(ValueError, match="asset IDs"):
        load_dataset(path)
    report = audit_training_data(robot_data, SERVING)
    assert not report["ready_for_candidate_comparison"]
    assert any(issue["code"] == "invalid_dataset" and issue["split"] == "train" for issue in report["issues"])


@pytest.mark.parametrize("groups", [np.arange(108), np.full(108, np.nan)])
def test_numeric_or_nan_asset_arrays_are_not_coerced_into_robot_ids(robot_data, groups):
    path = robot_data / "enhanced_train.npz"
    replace_arrays(path, groups=groups)
    with pytest.raises(ValueError, match="stored as strings"):
        load_dataset(path, with_features=False)


@pytest.mark.parametrize("changes", [
    {"X": np.zeros((108, 29, 7))}, {"S": np.zeros((108, 8))},
    {"y": np.zeros(108, dtype=float)}, {"groups": np.array(["robot"])},
    {"X": np.full((108, 30, 7), np.nan)}, {"S": np.full((108, 9), np.inf)},
    {"X": np.full((108, 30, 7), 1e100)}, {"S": np.full((108, 9), "not-numeric")},
    {"y": np.array(0)},
])
def test_audit_reports_invalid_sensor_or_label_arrays(robot_data, changes):
    replace_arrays(robot_data / "enhanced_val.npz", **changes)
    report = audit_training_data(robot_data, SERVING)
    assert not report["ready_for_candidate_comparison"]
    assert any(issue["code"] == "invalid_dataset" and issue["split"] == "validation" for issue in report["issues"])


@pytest.mark.parametrize("unknown", [-1, 9])
def test_audit_rejects_unknown_labels(robot_data, unknown):
    with np.load(robot_data / "enhanced_train.npz") as source:
        labels = source["y"]
    labels[0] = unknown
    replace_arrays(robot_data / "enhanced_train.npz", y=labels)
    report = audit_training_data(robot_data, SERVING)
    assert any("unknown train class" in issue["detail"] for issue in report["issues"])


def test_missing_validation_class_blocks_complete_macro_f1_comparison(robot_data, tmp_path, monkeypatch):
    import train_diagnosis

    def unexpected(*args, **kwargs):
        pytest.fail("missing validation classes must stop before model fitting")

    monkeypatch.setattr(train_diagnosis.lgb.LGBMClassifier, "fit", unexpected)
    with np.load(robot_data / "enhanced_val.npz") as source:
        labels = source["y"]
    labels[labels == 0] = 8
    replace_arrays(robot_data / "enhanced_val.npz", y=labels)
    report = audit_training_data(robot_data, SERVING)
    assert not report["ready_for_candidate_comparison"]
    assert report["splits"]["validation"]["per_class"]["E-ENV-C"]["rows"] == 0
    assert any("missing declared diagnosis classes" in issue["detail"] for issue in report["issues"])
    with pytest.raises(ValueError, match="validation is missing"):
        train_candidates(robot_data, tmp_path / "not-started", SERVING, rounds=2, data_kind="synthetic_fixture")
    assert not (tmp_path / "not-started").exists()


def test_many_windows_from_one_robot_do_not_count_as_independent_fault_assets(robot_data):
    with np.load(robot_data / "enhanced_train.npz") as source:
        labels, groups = source["y"], source["groups"]
    labels[(labels == 0) & (groups != "train-0")] = 8
    replace_arrays(robot_data / "enhanced_train.npz", y=labels)
    report = audit_training_data(robot_data, SERVING)
    assert report["splits"]["train"]["per_class"]["E-ENV-C"] == {"rows": 2, "assets": 1}
    assert any(issue["code"] == "holdout_unavailable" for issue in report["issues"])
    assert not report["ready_for_candidate_comparison"]


def test_official_split_asset_overlap_is_reported(robot_data):
    with np.load(robot_data / "enhanced_val.npz") as source:
        groups = source["groups"].astype("U16")
    groups[groups == "val-0"] = "train-0"
    replace_arrays(robot_data / "enhanced_val.npz", groups=groups)
    report = audit_training_data(robot_data, SERVING)
    assert report["official_asset_overlap"] == {"count": 1, "examples": ["train-0"]}
    assert any(issue["code"] == "official_asset_overlap" for issue in report["issues"])


@pytest.mark.parametrize("metadata", [[], {"err_map": {"normal": True}}])
def test_invalid_metadata_is_a_structured_issue(robot_data, metadata):
    (robot_data / "enhanced_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    report = audit_training_data(robot_data, SERVING)
    assert any(issue["code"] == "invalid_metadata" for issue in report["issues"])


@pytest.mark.parametrize("change", ["bool_index", "duplicate_index", "missing_normal"])
def test_invalid_class_mapping_is_reported(robot_data, change):
    path = robot_data / "enhanced_meta.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if change == "bool_index":
        metadata["err_map"]["E-ENV-C"] = False
    elif change == "duplicate_index":
        metadata["err_map"]["E-ENV-O"] = 0
    else:
        metadata["err_map"].pop("정상")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    report = audit_training_data(robot_data, SERVING)
    assert not report["ready_for_candidate_comparison"]
    assert any("class mapping" in issue["detail"] for issue in report["issues"])


def test_corrupt_npz_is_a_structured_issue(robot_data):
    (robot_data / "enhanced_train.npz").write_bytes(b"not an npz")
    report = audit_training_data(robot_data, SERVING)
    assert any(issue["code"] == "invalid_dataset" for issue in report["issues"])


def test_cli_saves_readiness_json_and_refuses_to_overwrite(robot_data, tmp_path, capsys):
    report_path = tmp_path / "audit.json"
    args = ["--data-dir", str(robot_data), "--champion-dir", str(SERVING), "--preflight-only",
            "--preflight-report", str(report_path), "--data-kind", "synthetic_fixture"]
    assert main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert json.loads(report_path.read_text(encoding="utf-8")) == stdout
    assert stdout["ready_for_candidate_comparison"]
    before = report_path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert report_path.read_bytes() == before


def test_cli_returns_blocked_status_but_keeps_missing_data_report(tmp_path, capsys):
    path = tmp_path / "blocked.json"
    assert main(["--data-dir", str(tmp_path / "absent"), "--preflight-only",
                 "--preflight-report", str(path)]) == 2
    assert not json.loads(capsys.readouterr().out)["ready_for_candidate_comparison"]
    assert path.is_file()


def test_report_option_requires_preflight_mode():
    with pytest.raises(SystemExit) as error:
        main(["--preflight-report", "/tmp/unused-audit.json"])
    assert error.value.code == 2
