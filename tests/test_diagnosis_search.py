"""Controlled sampling comparisons and evidence retention; no field-performance claims."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnosis_evaluation import asset_holdout
from train_diagnosis import dataset_support, file_sha256, main, train_candidates, training_rows


@pytest.mark.parametrize("ratio,normal_count", [(None, 120), (2, 12), (3, 18), (5, 30)])
def test_sampling_caps_only_normals_inside_fit(ratio, normal_count):
    labels = np.array([0] * 3 + [1] * 3 + [2] * 120 + [0, 1, 2])
    fit = np.arange(126)
    selected = training_rows(labels, fit, normal=2, seed=42, normal_ratio=ratio)
    assert np.sum(labels[selected] == 2) == normal_count
    assert set(range(6)) <= set(selected)
    assert not set(selected) & {126, 127, 128}
    np.testing.assert_array_equal(selected, training_rows(labels, fit, 2, 42, ratio))
    assert len(set(selected)) == len(selected)
    np.testing.assert_array_equal(fit, np.arange(126))


@pytest.mark.parametrize("ratio", [0, -1, 1.5, True])
def test_sampling_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="ratio"):
        training_rows(np.array([0, 1]), np.arange(2), 1, 42, ratio)


def test_sampling_never_upsamples():
    labels = np.array([0, 0, 1, 1, 1])
    np.testing.assert_array_equal(training_rows(labels, np.arange(5), 1, 42, 5), np.arange(5))


def test_all_class_holdout_rejects_a_single_fault_asset():
    labels = np.array([0, 1, 0, 0])
    groups = np.array(["a", "a", "b", "c"])
    with pytest.raises(ValueError, match="at least 2 independent assets"):
        asset_holdout(labels, groups, require_all_classes=True)


def test_all_class_holdout_covers_both_sides_without_asset_overlap():
    labels = np.tile([0, 1, 2], 12)
    groups = np.repeat(np.arange(12), 3)
    fit, selection = asset_holdout(labels, groups, require_all_classes=True)
    assert set(labels[fit]) == set(labels[selection]) == {0, 1, 2}
    assert not set(groups[fit]) & set(groups[selection])


def test_support_counts_windows_and_independent_assets_separately():
    support = dataset_support(np.array([0, 0, 1]), np.array(["a", "a", "b"]), ["normal", "fault", "absent"])
    assert support == {"normal": {"rows": 2, "assets": 1}, "fault": {"rows": 1, "assets": 1},
                       "absent": {"rows": 0, "assets": 0}}


@pytest.mark.parametrize("config", [
    {"normal_ratios": ()}, {"normal_ratios": (2, 2)}, {"normal_ratios": (-1,)},
    {"weight_modes": ()}, {"weight_modes": ("invalid",)},
    {"weight_modes": ("balanced", "balanced")}, {"data_kind": "physical_robot"},
    {"rounds": 0}, {"max_false_alarm": float("nan")},
])
def test_search_rejects_invalid_config_before_loading_data(tmp_path, config):
    with pytest.raises(ValueError):
        train_candidates(tmp_path / "missing", tmp_path / "output", ROOT / "data/processed", **config)
    assert not (tmp_path / "output").exists()


def test_search_protects_serving_directory_and_its_children(tmp_path):
    serving = tmp_path / "serving"
    for destination in (serving, serving / "candidate", tmp_path):
        with pytest.raises(ValueError, match="overwrite"):
            train_candidates(tmp_path / "missing", destination, serving)


def test_cli_maps_zero_to_full_data_and_forwards_grid(tmp_path, monkeypatch, capsys):
    import train_diagnosis
    received = {}

    def fake_train(*args):
        received["ratios"], received["modes"], received["kind"] = args[6:9]
        return {"selected_candidate": "test", "candidate": {}, "promotion": {"eligible": False}}

    monkeypatch.setattr(train_diagnosis, "train_candidates", fake_train)
    assert main(["--output", str(tmp_path), "--normal-ratios", "0", "2", "5",
                 "--weight-modes", "unweighted", "balanced", "--data-kind", "synthetic_fixture"]) == 0
    assert received == {"ratios": (None, 2, 5), "modes": ("unweighted", "balanced"), "kind": "synthetic_fixture"}
    assert json.loads(capsys.readouterr().out)["data_kind"] == "synthetic_fixture"


def test_file_digest_uses_file_contents(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
