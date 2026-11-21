"""Support policy tests on synthetic arrays; no real model-performance evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import audit_diagnosis_holdout as inspector
from diagnosis_evaluation import class_support_audit, validate_support_minima
import train_diagnosis as trainer


def supported_rows():
    return np.repeat([0, 1], 20), np.tile(np.repeat(["a", "b"], 10), 2), ["fault", "정상"]


def test_exact_boundaries_counts_and_input_preservation():
    y, groups, names = supported_rows()
    before_y, before_groups = y.copy(), groups.copy()
    report = class_support_audit(y, groups, names, min_rows_per_class=20, min_assets_per_class=2)
    assert report["sufficient_support"] and not report["issues"]
    assert report["per_class"] == {name: {"rows": 20, "assets": 2} for name in names}
    assert report["policy"] == {"min_rows_per_class": 20, "min_assets_per_class": 2}
    assert report["rows"] == 40 and report["assets"] == 2
    assert report["policy_basis"] == "declared_engineering_minima_not_statistical_power"
    np.testing.assert_array_equal(y, before_y)
    np.testing.assert_array_equal(groups, before_groups)
    json.dumps(report, allow_nan=False)


def test_many_windows_on_one_robot_do_not_inflate_asset_support():
    y, _, names = supported_rows()
    report = class_support_audit(y, np.full(len(y), "same"), names,
                                 min_rows_per_class=20, min_assets_per_class=2)
    assert not report["sufficient_support"]
    assert report["issues"] == [{"code": "insufficient_class_assets", "class_name": name,
                                  "actual": 1, "minimum": 2} for name in names]


@pytest.mark.parametrize("rows,assets,code", [(21, 2, "insufficient_class_rows"), (20, 3, "insufficient_class_assets")])
def test_one_below_configured_minimum_is_not_rounded_up(rows, assets, code):
    report = class_support_audit(*supported_rows(), min_rows_per_class=rows, min_assets_per_class=assets)
    assert not report["sufficient_support"] and [i["code"] for i in report["issues"]] == [code, code]


def test_missing_and_empty_classes_keep_zero_counts_and_fail_both_minima():
    for labels, groups in ((np.array([1]), np.array(["a"])),
                           (np.array([], dtype=int), np.array([], dtype=str))):
        report = class_support_audit(labels, groups, ["fault", "정상"])
        assert report["per_class"]["fault"] == {"rows": 0, "assets": 0}
        assert {i["code"] for i in report["issues"] if i["class_name"] == "fault"} == {
            "insufficient_class_rows", "insufficient_class_assets"}
        assert not report["sufficient_support"]


@pytest.mark.parametrize("invalid", [0, -1, True, 2.5, "2", None, float("inf")])
@pytest.mark.parametrize("field", ["min_rows_per_class", "min_assets_per_class"])
def test_invalid_minima_rejected(invalid, field):
    with pytest.raises(ValueError, match="positive integers"):
        class_support_audit(*supported_rows(), **{field: invalid})


def test_numpy_integer_minima_normalized_to_json_safe_python_integers():
    policy = validate_support_minima(np.int64(20), np.int32(2))
    assert all(type(value) is int for value in policy.values())


@pytest.mark.parametrize("labels,groups", [
    (np.array([0.0]), np.array(["a"])), (np.array([True]), np.array(["a"])),
    (np.array([-1]), np.array(["a"])), (np.array([2]), np.array(["a"])),
    (np.array([[0]]), np.array(["a"])), (np.array([0]), np.array(["a", "b"])),
    (np.array([0]), np.array([1])), (np.array([0]), np.array([None])),
    (np.array([0]), np.array([" a"])), (np.array([0]), np.array(["NaN"])),
])
def test_invalid_arrays_are_not_counted_as_valid_support(labels, groups):
    with pytest.raises(ValueError):
        class_support_audit(labels, groups, ["fault", "정상"])


@pytest.mark.parametrize("names", [["fault"], ["fault", "fault"], ["fault", "normal"], ["fault", " 正常 "]])
def test_ambiguous_class_names_rejected(names):
    with pytest.raises(ValueError, match="class names"):
        class_support_audit(np.array([0]), np.array(["a"]), names)


@pytest.fixture
def inputs(tmp_path):
    data, champion = tmp_path / "data", tmp_path / "champion"
    data.mkdir()
    champion.mkdir()
    metadata = json.loads((ROOT / "data/processed/enhanced_meta.json").read_text())
    for folder in (data, champion):
        (folder / "enhanced_meta.json").write_text(json.dumps(metadata))
    names = [name for name, _ in sorted(metadata["err_map"].items(), key=lambda item: item[1])]
    (champion / "robot_pdm_enhanced_meta.json").write_text(json.dumps({"class_names": names, "n_features": 249}))
    (champion / "robot_pdm_enhanced.txt").write_bytes(b"synthetic placeholder; preflight checks presence only")
    y = np.tile(np.repeat(np.arange(len(names)), 4), 6)
    for split in ("train", "val"):
        np.savez_compressed(data / f"enhanced_{split}.npz", X=np.zeros((len(y), 30, 7), dtype=np.float32),
                            S=np.zeros((len(y), 9), dtype=np.float32), y=y,
                            groups=np.repeat([f"{split}-{i}" for i in range(6)], len(names) * 4))
    return data, champion


def test_audit_reads_only_training_without_features_models_or_input_changes(inputs, monkeypatch):
    data, champion = inputs
    (data / "enhanced_val.npz").write_bytes(b"not a usable validation file")
    (champion / "robot_pdm_enhanced.txt").write_bytes(b"not a usable saved model")
    before = {p: p.read_bytes() for folder in inputs for p in folder.iterdir()}
    loaded, original = [], inspector.load_dataset

    def track(path, **kwargs):
        assert path.name == "enhanced_train.npz" and kwargs == {"with_features": False, "keep_raw": False}
        loaded.append(path.name)
        return original(path, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("support audit must not create features, fit, or load a Booster")

    monkeypatch.setattr(inspector, "load_dataset", track)
    monkeypatch.setattr(trainer, "make_feature_matrix_batched", forbidden)
    monkeypatch.setattr(trainer.lgb.LGBMClassifier, "fit", forbidden)
    monkeypatch.setattr(trainer, "load_booster", forbidden)
    report = inspector.audit_holdout(data, champion, min_selection_rows=8, min_selection_assets=2)
    assert report["support_gate_passed"] and not report["issues"]
    assert loaded == ["enhanced_train.npz"] and not report["official_validation_read"]
    assert not report["feature_matrix_created"] and not report["raw_sensor_arrays_retained"]
    assert not report["model_trained"] and not report["model_performance_evaluated"]
    assert set(report["lineage"]["sha256"]) == {"enhanced_train.npz", "enhanced_meta.json"}
    assert report["holdout"]["asset_overlap"] == 0
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("selection_only", [False, True])
def test_insufficient_support_stops_actual_training_path_before_fit_or_output(inputs, tmp_path, monkeypatch, selection_only):
    data, champion = inputs
    monkeypatch.setattr(trainer.lgb.LGBMClassifier, "fit", lambda *a, **k: pytest.fail("must gate before fitting"))
    output = tmp_path / "blocked-training"
    with pytest.raises(ValueError, match="insufficient selection support"):
        trainer.train_candidates(data, output, champion, rounds=1, selection_only=selection_only,
                                  min_selection_rows=9, min_selection_assets=2)
    assert not output.exists()


def test_default_training_behavior_preserved_and_policy_persisted(inputs, tmp_path):
    data, champion = inputs
    output = tmp_path / "candidate"
    report = trainer.train_candidates(data, output, champion, rounds=2, normal_ratios=(None,),
        weight_modes=("unweighted",), selection_only=True, data_kind="synthetic_fixture")
    support = report["selection_support_audit"]
    assert support["sufficient_support"] and support["policy"] == {"min_rows_per_class": 1, "min_assets_per_class": 1}
    saved = json.loads((output / "robot_pdm_enhanced_meta.json").read_text())
    events = [json.loads(line) for line in (output / "training_events.jsonl").read_text().splitlines()]
    assert saved["selection_support_policy"] == support["policy"]
    assert events[0]["selection_support_audit"] == support


@pytest.mark.parametrize("minimum,passes", [(8, True), (9, False)])
def test_training_cli_forwards_explicit_policy(inputs, tmp_path, minimum, passes):
    data, champion = inputs
    output = tmp_path / "cli-training"
    args = ["--data-dir", str(data), "--champion-dir", str(champion), "--output", str(output),
            "--selection-only", "--rounds", "1", "--normal-ratios", "0", "--weight-modes", "unweighted",
            "--min-selection-rows", str(minimum), "--min-selection-assets", "2",
            "--data-kind", "synthetic_fixture"]
    if passes:
        assert trainer.main(args) == 0
        report = json.loads((output / "experiment_report.json").read_text())
        assert report["selection_support_audit"]["policy"] == {"min_rows_per_class": 8, "min_assets_per_class": 2}
    else:
        with pytest.raises(SystemExit) as error:
            trainer.main(args)
        assert error.value.code == 2 and not output.exists()


def test_support_cli_passes_explicit_boundary_policy_and_saves_it(inputs, tmp_path):
    data, champion = inputs
    output = tmp_path / "passing.json"
    assert inspector.main(["--data-dir", str(data), "--champion-dir", str(champion), "--report", str(output),
                           "--min-selection-rows", "8", "--min-selection-assets", "2"]) == 0
    report = json.loads(output.read_text())
    assert report["support_gate_passed"] and report["selection_support_policy"] == {
        "min_rows_per_class": 8, "min_assets_per_class": 2}


def test_full_preflight_and_training_only_audit_share_policy_and_counts(inputs):
    data, champion = inputs
    full = trainer.audit_training_data(data, champion, min_selection_rows=9, min_selection_assets=2)
    only = inspector.audit_holdout(data, champion, min_selection_rows=9, min_selection_assets=2)
    assert not full["ready_for_candidate_comparison"] and not only["support_gate_passed"]
    assert full["selection_plan"]["support_audit"] == only["selection_support_audit"]
    assert full["issues"][0]["code"] == "insufficient_selection_support"


def test_cli_saves_blocked_report_and_protects_existing_or_dangling_output(inputs, tmp_path, monkeypatch, capsys):
    data, champion = inputs
    output = tmp_path / "blocked.json"
    args = ["--data-dir", str(data), "--champion-dir", str(champion), "--report", str(output)]
    assert inspector.main(args) == 2
    before = output.read_bytes()
    assert json.loads(before) == json.loads(capsys.readouterr().out)
    assert not json.loads(before)["support_gate_passed"]
    monkeypatch.setattr(inspector, "audit_holdout", lambda *a, **k: pytest.fail("check output before reading input"))
    with pytest.raises(SystemExit):
        inspector.main(args)
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(SystemExit):
        inspector.main([*args[:-1], str(link)])
    assert output.read_bytes() == before and not (tmp_path / "absent").exists()


@pytest.mark.parametrize("kind", ["missing_class", "single_asset_class"])
def test_unavailable_complete_holdout_is_reported_without_inventing_support(inputs, kind):
    data, champion = inputs
    path = data / "enhanced_train.npz"
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    mask = arrays["y"] == 0
    if kind == "single_asset_class":
        mask &= arrays["groups"] != "train-0"
    arrays["y"][mask] = 8
    np.savez_compressed(path, **arrays)
    report = inspector.audit_holdout(data, champion)
    assert not report["support_gate_passed"] and report["selection_support_audit"] is None
    assert report["holdout"] is None and report["issues"][0]["code"] == "holdout_unavailable"


@pytest.mark.parametrize("entry", ["trainer", "inspector"])
def test_cli_invalid_policy_rejected_before_training_or_input(inputs, tmp_path, monkeypatch, entry):
    data, champion = inputs
    monkeypatch.setattr(trainer, "load_dataset", lambda *a, **k: pytest.fail("must reject policy first"))
    monkeypatch.setattr(inspector, "load_dataset", lambda *a, **k: pytest.fail("must reject policy first"))
    function = trainer.main if entry == "trainer" else inspector.main
    args = ["--data-dir", str(data), "--champion-dir", str(champion), "--min-selection-rows", "0"]
    if entry == "inspector":
        args += ["--report", str(tmp_path / "never-written.json")]
    with pytest.raises(SystemExit) as error:
        function(args)
    assert error.value.code == 2
