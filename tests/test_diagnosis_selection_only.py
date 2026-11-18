"""Training-only search on synthetic windows; no real-data performance claims."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_diagnosis


@pytest.fixture
def training_inputs(tmp_path):
    data, champion = tmp_path / "data", tmp_path / "champion"
    data.mkdir()
    champion.mkdir()
    metadata = json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())
    names = [name for name, _ in sorted(metadata["err_map"].items(), key=lambda pair: pair[1])]
    for folder in (data, champion):
        (folder / "enhanced_meta.json").write_text(json.dumps(metadata))
    (champion / "robot_pdm_enhanced_meta.json").write_text(json.dumps({"class_names": names, "n_features": 249}))
    normal = names.index("정상")
    per_asset = np.repeat(np.arange(len(names)), [40 if i == normal else 2 for i in range(len(names))])
    y = np.tile(per_asset, 6)
    raw = np.random.default_rng(7).normal(size=(len(y), 30, 7)).astype(np.float32)
    np.savez_compressed(data / "enhanced_train.npz", X=raw, S=np.zeros((len(y), 9), dtype=np.float32),
                        y=y, groups=np.repeat([f"train-{i}" for i in range(6)], len(per_asset)))
    return data, champion


@pytest.mark.parametrize("criterion", ["macro_f1", "constrained_macro_f1", "multi_logloss"])
def test_search_never_opens_validation_and_later_full_run_selects_the_same_model(
        training_inputs, tmp_path, monkeypatch, criterion):
    data, champion = training_inputs
    output = tmp_path / "selection"
    before = {path: path.read_bytes() for folder in (data, champion) for path in folder.iterdir()}
    loaded, hashed, predicted_rows = [], [], []
    original_load, original_hash = train_diagnosis.load_dataset, train_diagnosis.file_sha256
    original_predict = train_diagnosis.lgb.LGBMClassifier.predict_proba

    def track_load(path, **kwargs):
        assert path.name == "enhanced_train.npz"
        loaded.append(path.name)
        return original_load(path, **kwargs)

    def track_hash(path):
        assert path.name not in {"enhanced_val.npz", "robot_pdm_enhanced.txt"}
        hashed.append(path.name)
        return original_hash(path)

    def track_predictions(model, features, *args, **kwargs):
        predicted_rows.append(len(features))
        return original_predict(model, features, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("selection-only search must not load/evaluate a champion, analyze Validation, or run promotion review")

    with monkeypatch.context() as isolated:
        isolated.setattr(train_diagnosis, "load_dataset", track_load)
        isolated.setattr(train_diagnosis, "file_sha256", track_hash)
        isolated.setattr(train_diagnosis.lgb.LGBMClassifier, "predict_proba", track_predictions)
        isolated.setattr(train_diagnosis, "load_booster", forbidden)
        isolated.setattr(train_diagnosis, "promotion_gate", forbidden)
        isolated.setattr(train_diagnosis, "asset_error_analysis", forbidden)
        report = train_diagnosis.train_candidates(data, output, champion, rounds=2,
                                                   iteration_metric=criterion, selection_only=True,
                                                   data_kind="synthetic_fixture")
    assert loaded == ["enhanced_train.npz"]
    assert set(hashed) == {"enhanced_train.npz", "enhanced_meta.json", "train_diagnosis.py",
                           "diagnosis_evaluation.py", "pdm_runtime.py"}
    assert report["selection_only"] and not report["official_validation_evaluated"]
    assert report["schema"] == "servicerobot.diagnosis.selection.v1"
    assert report["evaluation_scope"] == "training_robot_holdout_only"
    assert report["candidate"] is None and report["champion"] is None
    assert report["dataset_support"]["official_validation"] is None
    assert report["split_audit"]["official_validation_assets"] is None
    assert report["split_audit"]["asset_overlap"] is None
    assert not report["split_audit"]["official_asset_overlap_checked"]
    assert set(report["lineage"]["sha256"]) == {"enhanced_train.npz", "enhanced_meta.json"}
    assert report["lineage"]["champion_sha256"] is None
    assert not report["promotion"]["eligible"] and not report["promotion"]["automatic_promotion"]
    assert "official_validation_not_evaluated" in report["promotion"]["reasons"]
    winner = next(row for row in report["experiments"] if row["candidate_id"] == report["selected_candidate"])
    assert report["selected_holdout_metrics"] == winner["selection_metrics"]
    selection_rows = sum(row["rows"] for row in report["dataset_support"]["selection"].values())
    assert predicted_rows == [selection_rows] * 12
    assert len(pd.read_csv(output / "selection_results.csv")) == len(report["experiments"]) == 12
    assert json.loads((output / "experiment_report.json").read_text()) == report
    saved_meta = json.loads((output / "robot_pdm_enhanced_meta.json").read_text())
    assert not {"val_acc", "val_macro_f1"} & set(saved_meta)
    assert saved_meta["evaluation_scope"] == report["evaluation_scope"]
    assert not saved_meta["official_validation_evaluated"] and not saved_meta["automatic_promotion"]
    events = [json.loads(line) for line in (output / "training_events.jsonl").read_text().splitlines()]
    assert events[0]["selection_only"] and not events[0]["official_validation_planned"]
    assert events[0]["evaluation_scope"] == report["evaluation_scope"]
    assert events[-1]["event"] == "run_completed" and not events[-1]["official_validation_evaluated"]
    assert not (data / "enhanced_val.npz").exists()
    assert not (champion / "robot_pdm_enhanced.txt").exists()
    assert {path: path.read_bytes() for path in before} == before

    # Validation and a synthetic comparison model only become available after selection completes.
    with np.load(data / "enhanced_train.npz", allow_pickle=False) as source:
        validation = {key: source[key].copy() for key in source.files}
    validation["X"] += .1
    validation["groups"] = np.array([group.replace("train-", "val-") for group in validation["groups"]])
    np.savez_compressed(data / "enhanced_val.npz", **validation)
    selected_model = (output / "robot_pdm_enhanced.txt").read_bytes()
    (champion / "robot_pdm_enhanced.txt").write_bytes(selected_model)
    full = tmp_path / "full"
    evaluated = train_diagnosis.train_candidates(data, full, champion, rounds=2,
                                                  iteration_metric=criterion, data_kind="synthetic_fixture")
    assert not evaluated["selection_only"] and evaluated["official_validation_evaluated"]
    assert evaluated["schema"] == "servicerobot.diagnosis.experiment.v2"
    assert evaluated["evaluation_scope"] == "official_validation_comparison"
    assert evaluated["candidate"] is not None and evaluated["champion"] is not None
    assert evaluated["split_audit"]["official_asset_overlap_checked"]
    assert evaluated["split_audit"]["asset_overlap"] == 0
    assert evaluated["selected_candidate"] == report["selected_candidate"]
    assert evaluated["selected_holdout_metrics"] == report["selected_holdout_metrics"]
    assert [row["selection_metrics"] for row in evaluated["experiments"]] == [
        row["selection_metrics"] for row in report["experiments"]]
    assert (full / "robot_pdm_enhanced.txt").read_bytes() == selected_model
    evaluated_meta = json.loads((full / "robot_pdm_enhanced_meta.json").read_text())
    assert evaluated_meta["val_macro_f1"] == evaluated["candidate"]["macro_f1"]
    assert evaluated_meta["val_acc"] == evaluated["candidate"]["accuracy"]
    assert not evaluated["promotion"]["automatic_promotion"]
    assert (champion / "robot_pdm_enhanced.txt").read_bytes() == selected_model


def test_selection_only_ignores_even_present_but_invalid_validation_and_champion_files(training_inputs, tmp_path):
    data, champion = training_inputs
    validation, model = data / "enhanced_val.npz", champion / "robot_pdm_enhanced.txt"
    validation.write_bytes(b"not a usable NPZ")
    model.write_bytes(b"not a usable model")
    report = train_diagnosis.train_candidates(data, tmp_path / "selection", champion, rounds=1,
                                               normal_ratios=(3,), weight_modes=("unweighted",),
                                               selection_only=True, data_kind="synthetic_fixture")
    assert not report["official_validation_evaluated"]
    assert validation.read_bytes() == b"not a usable NPZ"
    assert model.read_bytes() == b"not a usable model"
    with pytest.raises((ValueError, OSError)):
        train_diagnosis.train_candidates(data, tmp_path / "full", champion, rounds=1)
    assert not (tmp_path / "full").exists()


def test_caller_declared_aihub_does_not_enable_promotion_without_official_evaluation(training_inputs, tmp_path):
    data, champion = training_inputs
    report = train_diagnosis.train_candidates(data, tmp_path / "declared-source", champion, rounds=1,
                                               normal_ratios=(3,), weight_modes=("unweighted",),
                                               selection_only=True, data_kind="aihub_recorded")
    assert report["candidate"] is None and report["champion"] is None
    assert not report["promotion"]["eligible"]
    assert "official_validation_not_evaluated" in report["promotion"]["reasons"]


def test_selected_artifact_can_be_evaluated_later_without_rewriting_training_only_evidence(training_inputs, tmp_path):
    from evaluate_enhanced import evaluate_saved_model

    data, champion = training_inputs
    output = tmp_path / "selected"
    report = train_diagnosis.train_candidates(data, output, champion, rounds=1,
                                               normal_ratios=(3,), weight_modes=("unweighted",),
                                               selection_only=True, data_kind="synthetic_fixture")
    with np.load(data / "enhanced_train.npz", allow_pickle=False) as source:
        validation = {key: source[key].copy() for key in source.files}
    validation["groups"] = np.array([group.replace("train-", "val-") for group in validation["groups"]])
    np.savez_compressed(data / "enhanced_val.npz", **validation)
    before = {path: path.read_bytes() for folder in (data, output) for path in folder.iterdir()}
    evaluated = evaluate_saved_model(data, output, "synthetic_fixture")
    assert evaluated["rows"] == len(validation["y"])
    assert evaluated["class_coverage"] == 9
    assert evaluated["evaluation"]["data_kind"] == "synthetic_fixture"
    assert not evaluated["evaluation"]["automatic_promotion"]
    assert json.loads((output / "experiment_report.json").read_text()) == report
    assert not report["official_validation_evaluated"]
    assert {path: path.read_bytes() for path in before} == before
