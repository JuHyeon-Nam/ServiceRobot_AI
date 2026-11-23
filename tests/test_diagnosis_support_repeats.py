"""Repeated support checks on synthetic windows, never performance or confidence intervals."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import audit_diagnosis_holdout as inspector
import train_diagnosis as trainer


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
    y = np.tile(np.repeat(np.arange(len(names)), 4), 6)
    np.savez_compressed(data / "enhanced_train.npz", X=np.zeros((len(y), 30, 7), dtype=np.float32),
                        S=np.zeros((len(y), 9), dtype=np.float32), y=y,
                        groups=np.repeat([f"robot-{i}" for i in range(6)], len(names) * 4))
    return data, champion


def test_single_and_repeated_results_agree_without_ranking_or_reloading(inputs, monkeypatch):
    data, champion = inputs
    (data / "enhanced_val.npz").write_bytes(b"not official data")
    (champion / "robot_pdm_enhanced.txt").write_bytes(b"not a Booster")
    before = {p: p.read_bytes() for folder in inputs for p in folder.iterdir()}
    calls, original = [], inspector.load_dataset

    def tracked(path, **kwargs):
        assert path.name == "enhanced_train.npz"
        assert kwargs == {"with_features": False, "keep_raw": False}
        calls.append(path.name)
        return original(path, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("repeats must not compute features, fit models, or evaluate a saved model")

    monkeypatch.setattr(inspector, "load_dataset", tracked)
    monkeypatch.setattr(trainer, "make_feature_matrix_batched", forbidden)
    monkeypatch.setattr(trainer.lgb.LGBMClassifier, "fit", forbidden)
    monkeypatch.setattr(trainer, "load_booster", forbidden)
    report = inspector.audit_repeated_holdouts(data, champion, seeds=[42, 43, 44],
                                               min_selection_rows=8, min_selection_assets=2)
    assert calls == ["enhanced_train.npz"]
    assert report["support_gate_passed"] and report["passing_request_count"] == 3
    assert report["requested_seed_count"] == report["generated_holdout_count"] == 3
    assert report["unavailable_holdout_count"] == 0 and not report["best_seed_selected"]
    assert not report["official_validation_read"] and not report["model_performance_evaluated"]
    assert report["schema"] == "servicerobot.diagnosis.holdout_support_repeats.v1"
    for run in report["runs"]:
        single = inspector.audit_holdout(data, champion, seed=run["seed"], min_selection_rows=8, min_selection_assets=2)
        for field in ("selection_support_audit", "holdout", "support_gate_passed", "issues"):
            assert run[field] == single[field]
    assert all(value == {"rows_min": 8, "rows_max": 8, "assets_min": 2, "assets_max": 2,
                         "evaluated_requests": 3, "requests_passing_minima": 3}
               for value in report["selection_support_ranges"].values())
    assert {p: p.read_bytes() for p in before} == before


def test_distinct_starting_seeds_can_repeat_one_partition_and_are_reported(inputs, monkeypatch):
    data, champion = inputs
    original = inspector.asset_holdout
    requested = []

    def fixed(labels, groups, seed, **kwargs):
        requested.append(seed)
        return original(labels, groups, 42, **kwargs)

    monkeypatch.setattr(inspector, "asset_holdout", fixed)
    report = inspector.audit_repeated_holdouts(data, champion, seeds=[42, 43, 44], min_selection_rows=8)
    assert requested == [42, 43, 44]
    assert report["distinct_selection_asset_sets"] == 1
    assert report["partition_requests"][0]["starting_seeds"] == [42, 43, 44]
    assert not report["best_seed_selected"]
    report["runs"][0]["issues"].append({"test": "mutated"})
    assert not report["runs"][1]["issues"]


def test_unavailable_requests_are_retained_and_excluded_from_range_denominator(inputs, monkeypatch):
    data, champion = inputs
    original = inspector.asset_holdout

    def sometimes_unavailable(labels, groups, seed, **kwargs):
        if seed == 43:
            raise ValueError("simulated unavailable class-complete holdout")
        return original(labels, groups, seed, **kwargs)

    monkeypatch.setattr(inspector, "asset_holdout", sometimes_unavailable)
    report = inspector.audit_repeated_holdouts(data, champion, seeds=[42, 43, 44], min_selection_rows=8)
    assert not report["support_gate_passed"]
    assert report["generated_holdout_count"] == 2 and report["unavailable_holdout_count"] == 1
    assert report["passing_request_count"] == 2
    unavailable = report["runs"][1]
    assert unavailable["seed"] == 43 and unavailable["holdout"] is None
    assert unavailable["selection_support_audit"] is None
    assert unavailable["issues"][0]["code"] == "holdout_unavailable"
    assert all(row["evaluated_requests"] == 2 for row in report["selection_support_ranges"].values())


def test_all_unavailable_requests_do_not_fabricate_zero_or_perfect_support(inputs, monkeypatch):
    data, champion = inputs

    def unavailable(*args, **kwargs):
        raise ValueError("unavailable")

    monkeypatch.setattr(inspector, "asset_holdout", unavailable)
    report = inspector.audit_repeated_holdouts(data, champion, seeds=[42, 43])
    assert not report["support_gate_passed"] and report["distinct_selection_asset_sets"] == 0
    assert report["generated_holdout_count"] == report["passing_request_count"] == 0
    assert report["unavailable_holdout_count"] == 2
    for row in report["selection_support_ranges"].values():
        assert row == {"rows_min": None, "rows_max": None, "assets_min": None, "assets_max": None,
                       "evaluated_requests": 0, "requests_passing_minima": 0}


@pytest.mark.parametrize("seeds", [[], [42, 42], [-1], [True], [1.5], ["42"], [None],
                                   [2**32 - 63], list(range(65)), "42"])
def test_invalid_requests_rejected_before_training_input_io(inputs, monkeypatch, seeds):
    data, champion = inputs
    monkeypatch.setattr(inspector, "_load_training_support", lambda *a: pytest.fail("reject seed configuration first"))
    with pytest.raises(ValueError):
        inspector.audit_repeated_holdouts(data, champion, seeds=seeds)


def test_numpy_starting_seed_normalized_to_python_integer(inputs):
    report = inspector.audit_repeated_holdouts(*inputs, seeds=[np.int64(42)])
    assert type(report["starting_seeds"][0]) is int
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("minimum,expected", [(8, 0), (9, 2)])
def test_repeated_cli_saves_every_request_and_returns_all_requests_policy(inputs, tmp_path, minimum, expected):
    data, champion = inputs
    output = tmp_path / "repeated.json"
    args = ["--data-dir", str(data), "--champion-dir", str(champion), "--seeds", "42", "43", "44",
            "--min-selection-rows", str(minimum), "--min-selection-assets", "2", "--report", str(output)]
    assert inspector.main(args) == expected
    report = json.loads(output.read_text())
    assert len(report["runs"]) == 3 and report["support_gate_passed"] == (expected == 0)
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        inspector.main(args)
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(SystemExit):
        inspector.main([*args[:-1], str(link)])
    assert output.read_bytes() == before and not (tmp_path / "absent").exists()


@pytest.mark.parametrize("seed", ["0", "42", "1000"])
@pytest.mark.parametrize("reverse", [False, True])
def test_cli_seed_modes_are_mutually_exclusive_before_input_io(tmp_path, monkeypatch, seed, reverse):
    def forbidden(*args, **kwargs):
        pytest.fail("conflicting seed options must fail before auditing inputs")

    monkeypatch.setattr(inspector, "audit_holdout", forbidden)
    monkeypatch.setattr(inspector, "audit_repeated_holdouts", forbidden)
    single, repeated = ["--seed", seed], ["--seeds", "43", "44"]
    modes = repeated + single if reverse else single + repeated
    output = tmp_path / "unused.json"
    with pytest.raises(SystemExit) as error:
        inspector.main(["--data-dir", str(tmp_path / "absent"), "--report", str(output), *modes])
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("options,expected_seed", [([], 42), (["--seed", "0"], 0),
                                                  (["--seed", "42"], 42), (["--seed", "1000"], 1000)])
def test_cli_applies_single_seed_default_after_parsing(tmp_path, monkeypatch, options, expected_seed):
    received = []

    def single(*args, **kwargs):
        received.append(kwargs["seed"])
        return {"support_gate_passed": True, "seed": kwargs["seed"]}

    def forbidden(*args, **kwargs):
        pytest.fail("single mode must not call repeated holdouts")

    monkeypatch.setattr(inspector, "audit_holdout", single)
    monkeypatch.setattr(inspector, "audit_repeated_holdouts", forbidden)
    output = tmp_path / "single.json"
    assert inspector.main(["--data-dir", str(tmp_path / "absent"), "--report", str(output), *options]) == 0
    assert received == [expected_seed]
    assert json.loads(output.read_text())["seed"] == expected_seed
