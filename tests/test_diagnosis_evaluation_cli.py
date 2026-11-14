"""Saved-model evaluation safety using synthetic windows and a temporary Booster."""
import hashlib
import gc
import json
from pathlib import Path
import sys
import weakref
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluate_enhanced
import pdm_runtime


@pytest.fixture
def saved_evaluation(tmp_path):
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    models.mkdir()
    metadata = json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())
    names = [name for name, _ in sorted(metadata["err_map"].items(), key=lambda pair: pair[1])]
    labels = np.tile(np.arange(len(names)), 4)
    raw = np.random.default_rng(7).normal(size=(len(labels), 30, 7)).astype(np.float32)
    raw[:, :, 0] += labels[:, None] * 10
    static = np.zeros((len(labels), 9), dtype=np.float32)
    groups = np.repeat(["validation-a", "validation-b", "validation-c", "validation-d"], len(names))
    np.savez(data / "enhanced_val.npz", X=raw, S=static, y=labels, groups=groups)
    for folder in (data, models):
        (folder / "enhanced_meta.json").write_text(json.dumps(metadata))
    (models / "robot_pdm_enhanced_meta.json").write_text(json.dumps({"class_names": names, "n_features": 249}))
    model = lgb.LGBMClassifier(n_estimators=2, num_leaves=4, min_child_samples=1,
                               objective="multiclass", verbosity=-1, n_jobs=2, random_state=7)
    model.fit(evaluate_enhanced.feat(raw, static), labels)
    model.booster_.save_model(str(models / "robot_pdm_enhanced.txt"))
    return data, models


def input_bytes(data, models):
    return {str(path): path.read_bytes() for folder in (data, models) for path in folder.iterdir()}


def test_saved_evaluation_records_counts_and_hashes_without_mutating_inputs(saved_evaluation):
    data, models = saved_evaluation
    before = input_bytes(data, models)
    report = evaluate_enhanced.evaluate_saved_model(data, models, "synthetic_fixture")
    evidence = report["evaluation"]
    assert evidence["schema"] == "servicerobot.diagnosis.evaluation.v1"
    assert evidence["data_kind"] == "synthetic_fixture"
    assert evidence["data_kind_source"] == "caller_declared_not_independently_verified"
    assert evidence["evaluated_at_utc"].endswith("+00:00")
    assert evidence["assets"] == 4
    assert evidence["dataset_file"] == "enhanced_val.npz"
    assert evidence["feature_construction"] == {
        "mode": "batched", "max_batch_rows": 4096, "dtype": "float32", "n_features": 249,
        "raw_sensor_arrays_retained": False}
    assert not evidence["automatic_promotion"]
    assert "training_validation_asset_overlap_not_checked" in evidence["limitations"]
    assert "source_authenticity_not_verified" in evidence["limitations"]
    assert report["rows"] == sum(map(sum, report["confusion_matrix"])) == 36
    assert report["class_coverage"] == len(report["class_order"]) == 9
    assert set(report["per_class"]) == set(report["class_order"])
    errors = report["asset_error_analysis"]
    assert errors["asset_count"] == evidence["assets"] == 4
    assert set(errors["per_asset"]) == {f"validation-{letter}" for letter in "abcd"}
    assert not errors["used_for_selection"]
    for key in ("rows", "accuracy", "normal_false_alarm_rate", "fault_miss_rate", "confusion_matrix"):
        assert errors["overall"][key] == report[key]
    np.testing.assert_array_equal(np.sum([row["confusion_matrix"] for row in errors["per_asset"].values()], axis=0),
                                  report["confusion_matrix"])
    assert evidence["per_class_support"] == {name: {"rows": 4, "assets": 4} for name in report["class_order"]}
    expected_files = {"validation_windows": data / "enhanced_val.npz",
                      "dataset_metadata": data / "enhanced_meta.json",
                      "model": models / "robot_pdm_enhanced.txt",
                      "model_metadata": models / "robot_pdm_enhanced_meta.json",
                      "model_encoding_metadata": models / "enhanced_meta.json"}
    assert evidence["input_sha256"] == {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for key, path in expected_files.items()}
    assert set(evidence["code_sha256"]) == {"evaluate_enhanced.py", "diagnosis_evaluation.py",
                                           "train_diagnosis.py", "pdm_runtime.py"}
    for name, digest in evidence["code_sha256"].items():
        assert digest == hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
    json.dumps(report, allow_nan=False)
    assert input_bytes(data, models) == before


def test_cli_report_matches_stdout_and_rejects_a_second_write(saved_evaluation, tmp_path, capsys):
    data, models = saved_evaluation
    target = tmp_path / "reports/evaluation.json"
    args = ["--data-dir", str(data), "--model-dir", str(models), "--report", str(target),
            "--data-kind", "synthetic_fixture"]
    assert evaluate_enhanced.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads(target.read_text()) == report
    assert report["evaluation"]["data_kind"] == "synthetic_fixture"
    before = target.read_bytes()
    with pytest.raises(SystemExit) as error:
        evaluate_enhanced.main(args)
    assert error.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert target.read_bytes() == before


def test_default_data_kind_is_unspecified(saved_evaluation):
    report = evaluate_enhanced.evaluate_saved_model(*saved_evaluation)
    assert report["evaluation"]["data_kind"] == "unspecified"


@pytest.mark.parametrize("kind", ["report", "model", "dataset", "directory", "symlink", "dangling_symlink"])
def test_output_guard_runs_before_evaluation(saved_evaluation, tmp_path, monkeypatch, capsys, kind):
    data, models = saved_evaluation
    target = tmp_path / "output.json"
    if kind == "report":
        target.write_text("keep existing result")
    elif kind == "model":
        target = models / "robot_pdm_enhanced.txt"
    elif kind == "dataset":
        target = data / "enhanced_val.npz"
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(models / ("robot_pdm_enhanced.txt" if kind == "symlink" else "missing.json"))
    before = input_bytes(data, models)
    target_before = target.read_bytes() if target.is_file() else None

    def forbidden_evaluation(*args):
        pytest.fail("existing output must be rejected before loading data or predicting")

    monkeypatch.setattr(evaluate_enhanced, "evaluate_saved_model", forbidden_evaluation)
    with pytest.raises(SystemExit) as error:
        evaluate_enhanced.main(["--report", str(target)])
    assert error.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert input_bytes(data, models) == before
    if target_before is not None:
        assert target.read_bytes() == target_before
    if "symlink" in kind:
        assert target.is_symlink()


def test_exclusive_create_handles_a_file_appearing_during_evaluation(saved_evaluation, tmp_path, monkeypatch, capsys):
    data, models = saved_evaluation
    target = tmp_path / "raced-report.json"
    before = input_bytes(data, models)
    original = evaluate_enhanced.evaluate_saved_model

    def concurrent_writer(*args):
        report = original(*args)
        target.write_text("concurrent result")
        return report

    monkeypatch.setattr(evaluate_enhanced, "evaluate_saved_model", concurrent_writer)
    with pytest.raises(SystemExit) as error:
        evaluate_enhanced.main(["--data-dir", str(data), "--model-dir", str(models), "--report", str(target)])
    assert error.value.code == 2
    assert "Diagnosis evaluation:" in capsys.readouterr().err
    assert target.read_text() == "concurrent result"
    assert input_bytes(data, models) == before


@pytest.mark.parametrize("target,key,value,message", [
    ("dataset", "devtype_map", {"different": 1}, "encoding contracts differ"),
    ("dataset", "mainstate_map", {"different": 1}, "encoding contracts differ"),
    ("dataset", "crowd_map", {"different": 1}, "encoding contracts differ"),
    ("dataset", "dyn", [], "encoding contracts differ"),
    ("dataset", "stat", [], "encoding contracts differ"),
    ("dataset", "err_map", {"정상": "0"}, "class mapping"),
    ("model", "n_features", 248, "class/feature contracts differ"),
    ("model", "class_names", ["정상", "fault"], "class/feature contracts differ"),
])
def test_contract_failures_stop_before_features_or_model_loading(saved_evaluation, tmp_path, monkeypatch,
                                                                capsys, target, key, value, message):
    data, models = saved_evaluation
    path = data / "enhanced_meta.json" if target == "dataset" else models / "robot_pdm_enhanced_meta.json"
    metadata = json.loads(path.read_text())
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    output = tmp_path / "invalid-report.json"

    def forbidden(*args):
        pytest.fail("invalid contract must stop before features or Booster loading")

    monkeypatch.setattr(evaluate_enhanced, "feat", forbidden)
    monkeypatch.setattr(evaluate_enhanced, "load_booster", forbidden)
    with pytest.raises(SystemExit) as error:
        evaluate_enhanced.main(["--data-dir", str(data), "--model-dir", str(models), "--report", str(output)])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("kind,message", [
    ("missing_class", "missing declared diagnosis classes"),
    ("unknown_class", "unknown validation class"),
    ("noninteger_labels", "class labels must be integer"),
    ("missing_groups", "dataset needs X/S/y/groups"),
    ("nonfinite_sensor", "must be finite"),
])
def test_bad_validation_windows_are_rejected_before_features(saved_evaluation, tmp_path, monkeypatch, capsys,
                                                             kind, message):
    data, models = saved_evaluation
    path = data / "enhanced_val.npz"
    with np.load(path) as arrays:
        rows = {key: arrays[key] for key in arrays.files}
    if kind == "missing_class":
        mask = rows["y"] != 0
        rows = {key: value[mask] for key, value in rows.items()}
    elif kind == "unknown_class":
        rows["y"][0] = 9
    elif kind == "noninteger_labels":
        rows["y"] = rows["y"].astype(float)
    elif kind == "missing_groups":
        rows.pop("groups")
    else:
        rows["X"][0, 0, 0] = np.nan
    np.savez(path, **rows)
    output = tmp_path / "invalid-data-report.json"

    def forbidden(*args):
        pytest.fail("invalid windows must stop before feature engineering")

    monkeypatch.setattr(evaluate_enhanced, "feat", forbidden)
    with pytest.raises(SystemExit) as error:
        evaluate_enhanced.main(["--data-dir", str(data), "--model-dir", str(models), "--report", str(output)])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("features,classes", [(248, 9), (249, 8)])
def test_booster_contract_is_checked_before_prediction(saved_evaluation, monkeypatch, features, classes):
    def forbidden_predict(*args):
        pytest.fail("incompatible Booster must not predict")

    booster = SimpleNamespace(num_feature=lambda: features, num_model_per_iteration=lambda: classes,
                              predict=forbidden_predict)
    monkeypatch.setattr(evaluate_enhanced, "load_booster", lambda path: booster)
    with pytest.raises(ValueError, match="saved booster and class/feature metadata differ"):
        evaluate_enhanced.evaluate_saved_model(*saved_evaluation)


def test_unknown_provenance_fails_before_loading_files(tmp_path):
    with pytest.raises(ValueError, match="unknown evaluation data kind"):
        evaluate_enhanced.evaluate_saved_model(tmp_path, tmp_path, "physical_robot")


def test_missing_input_is_a_cli_error_and_does_not_create_report(tmp_path, capsys):
    output = tmp_path / "missing-input-report.json"
    with pytest.raises(SystemExit) as error:
        evaluate_enhanced.main(["--data-dir", str(tmp_path), "--model-dir", str(tmp_path), "--report", str(output)])
    assert error.value.code == 2
    assert "Diagnosis evaluation:" in capsys.readouterr().err
    assert not output.exists()


def test_saved_evaluation_uses_bounded_feature_batches(saved_evaluation, monkeypatch):
    data, models = saved_evaluation
    path = data / "enhanced_val.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays = {key: np.concatenate([value] * 120, axis=0) for key, value in arrays.items()}
    np.savez_compressed(path, **arrays)
    before = input_bytes(data, models)
    original = pdm_runtime.make_feature_matrix
    calls = []

    def tracked(windows, contexts):
        calls.append(len(windows))
        return original(windows, contexts)

    monkeypatch.setattr(pdm_runtime, "make_feature_matrix", tracked)
    report = evaluate_enhanced.evaluate_saved_model(data, models, "synthetic_fixture")
    assert report["rows"] == 4320
    assert calls == [4096, 224]
    assert report["evaluation"]["feature_construction"]["max_batch_rows"] == 4096
    assert input_bytes(data, models) == before


def test_saved_evaluation_releases_raw_before_loading_booster(saved_evaluation, monkeypatch):
    data, models = saved_evaluation
    before = input_bytes(data, models)
    original_features, original_booster = evaluate_enhanced.feat, evaluate_enhanced.load_booster
    refs = []

    def track_features(raw, static):
        refs.extend([weakref.ref(raw), weakref.ref(static)])
        return original_features(raw, static)

    def check_before_booster(path):
        gc.collect()
        assert len(refs) == 2 and all(ref() is None for ref in refs)
        return original_booster(path)

    monkeypatch.setattr(evaluate_enhanced, "feat", track_features)
    monkeypatch.setattr(evaluate_enhanced, "load_booster", check_before_booster)
    report = evaluate_enhanced.evaluate_saved_model(data, models, "synthetic_fixture")
    assert report["rows"] == 36
    assert not report["evaluation"]["feature_construction"]["raw_sensor_arrays_retained"]
    assert input_bytes(data, models) == before
