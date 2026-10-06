"""Leakage, uncertainty, and serving evidence contracts for the ML upgrade."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_registry import load_report, summary
from cmapss_benchmark import (COLUMNS, causal_features, interval_radius, read_trajectories,
                              run_benchmark, split_units, truncated_indices)
from diagnosis_evaluation import asset_holdout, class_weights, classification_metrics, promotion_gate
from download_cmapss import prepare_archive
from pdm_runtime import make_feature_matrix, make_features
from train_diagnosis import load_dataset, train_candidates


def trajectory(units=20, cycles=40):
    rows = []
    for unit in range(1, units + 1):
        for cycle in range(1, cycles + 1):
            rows.append([unit, cycle, *[unit + cycle * i / 100 for i in range(1, 25)]])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_feature_batch_matches_legacy_and_online():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(3, 30, 7)).astype(np.float32)
    static = rng.normal(size=(3, 9)).astype(np.float32)
    x = raw[:, :, [0, 1, 4, 5, 6]]
    expected = np.c_[x.reshape(3, -1), x.mean(1), x.std(1),
                     x[:, -10:].mean(1) - x[:, :10].mean(1),
                     np.abs(np.fft.rfft(x, axis=1))[:, :15].reshape(3, -1), static].astype(np.float32)
    np.testing.assert_array_equal(make_feature_matrix(raw, static), expected)
    changed = raw.copy()
    changed[:, :, 2:4] += 10000
    np.testing.assert_array_equal(make_feature_matrix(changed, static), expected)
    context = dict(zip(["isOffline", "nowCharging", "emergencyStop", "batteryUse",
                        "batteryCycleCount", "distance"], static[0, :6]))
    context.update(crowd="MIDDLE", deviceType="robot", mainState="MOVE")
    meta = {"devtype_map": {"robot": 2}, "mainstate_map": {"MOVE": 3}}
    static[0, 6:] = [1, 2, 3]
    np.testing.assert_array_equal(make_features(raw[0].tolist(), context, meta),
                                  make_feature_matrix(raw[:1], static[:1]))


@pytest.mark.parametrize("shape", [(1, 29, 7), (1, 30, 6), (30, 7)])
def test_feature_shape_rejected(shape):
    with pytest.raises(ValueError):
        make_feature_matrix(np.zeros(shape), np.zeros((1, 9)))


def test_nonfinite_features_rejected():
    raw = np.zeros((1, 30, 7)); raw[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        make_feature_matrix(raw, np.zeros((1, 9)))


def test_robot_holdout_keeps_entire_assets():
    labels = np.tile([0, 1, 2], 12)
    groups = np.repeat(np.arange(12), 3)
    fit, held = asset_holdout(labels, groups)
    assert not set(groups[fit]) & set(groups[held])
    assert set(labels[fit]) == {0, 1, 2}


@pytest.mark.parametrize("groups", [["a", "b"], ["", "b", "c"]])
def test_robot_holdout_requires_independent_ids(groups):
    with pytest.raises(ValueError):
        asset_holdout(np.arange(len(groups)), np.array(groups))


def test_weights_raise_rare_class_without_resampling_validation():
    labels = np.array([0] * 100 + [1] * 2)
    weights = class_weights(labels, "balanced")
    assert weights[-1] > weights[0]
    assert weights.mean() == pytest.approx(1)
    np.testing.assert_array_equal(class_weights(labels, "unweighted"), np.ones(102))


def test_metrics_include_false_alarm_miss_and_calibration():
    metrics = classification_metrics(np.array([0, 0, 1, 1]),
                                     np.array([[.9, .1], [.1, .9], [.8, .2], [.2, .8]]), ["정상", "fault"])
    assert metrics["normal_false_alarm_rate"] == .5
    assert metrics["fault_miss_rate"] == .5
    assert metrics["accuracy"] == .5
    assert metrics["ece_10_bins"] > 0
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


@pytest.mark.parametrize("probabilities", [[[.5, .7]], [[np.nan, 1]], [[-.1, 1.1]]])
def test_invalid_probabilities_rejected(probabilities):
    with pytest.raises(ValueError):
        classification_metrics(np.array([0]), np.array(probabilities), ["정상", "fault"])


def test_promotion_rejects_accuracy_only_improvement():
    champion = dict(macro_f1=.6, accuracy=.9, normal_false_alarm_rate=.02,
                    fault_miss_rate=.2, class_coverage=2, class_order=["정상", "fault"])
    candidate = dict(champion, accuracy=.95, macro_f1=.59)
    assert not promotion_gate(champion, candidate)["eligible"]
    candidate.update(macro_f1=.65, fault_miss_rate=.1)
    gate = promotion_gate(champion, candidate)
    assert gate["eligible"] and not gate["automatic_promotion"]
    candidate["normal_false_alarm_rate"] = .1
    assert "normal_false_alarm_limit" in promotion_gate(champion, candidate)["reasons"]


def test_temporal_features_do_not_see_future_or_other_engines():
    frame = trajectory(2, 40)
    original = causal_features(frame, True)
    modified = frame.copy()
    modified.loc[modified.cycle > 20, COLUMNS[2:]] = 100000
    np.testing.assert_array_equal(original[frame.cycle <= 20],
                                  causal_features(modified, True)[frame.cycle <= 20])
    one_engine = causal_features(frame[frame.unit == 2], True)
    np.testing.assert_array_equal(original[frame.unit == 2], one_engine)


def test_engine_splits_and_truncation():
    frame = trajectory()
    fit, selection, calibration = split_units(frame, 42)
    cohorts = [set(frame.iloc[rows].unit) for rows in (fit, selection, calibration)]
    assert not cohorts[0] & cohorts[1] and not cohorts[1] & cohorts[2] and not cohorts[0] & cohorts[2]
    points = truncated_indices(frame, selection, 44)
    assert len(points) == len(cohorts[1])
    assert frame.iloc[points].cycle.between(16, 34).all()


@pytest.mark.parametrize("kind", ["nonfinite", "duplicate", "gap", "columns"])
def test_trajectory_parser_rejects_bad_data(tmp_path, kind):
    values = trajectory(1, 4).to_numpy()
    if kind == "nonfinite": values[0, 4] = np.nan
    if kind == "duplicate": values[1, :2] = values[0, :2]
    if kind == "gap": values[1, 1] = 10
    if kind == "columns": values = values[:, :-1]
    path = tmp_path / "bad.txt"; np.savetxt(path, values)
    with pytest.raises(ValueError): read_trajectories(path)


def test_conformal_finite_sample_rank():
    assert interval_radius(np.arange(1, 21), .9) == 19
    with pytest.raises(ValueError): interval_radius(np.array([1]), .9)
    with pytest.raises(ValueError): interval_radius(np.array([np.nan]), .9)


def test_benchmark_end_to_end_on_fixture(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    np.savetxt(data / "train_FD001.txt", trajectory(20, 40).to_numpy())
    np.savetxt(data / "test_FD001.txt", trajectory(3, 25).to_numpy())
    np.savetxt(data / "RUL_FD001.txt", [15, 20, 25])
    path = tmp_path / "report.json"
    report = run_benchmark(data, tmp_path / "models", path, rounds=2, coverage=.5)
    assert report["protocol"]["test_engines"] == 3
    assert [row["true_rul"] for row in report["engines"]] == [15, 20, 25]
    assert report["engines"][0]["history"][0]["true_rul"] == 39
    assert not report["serving_enabled_for_agv"]
    assert load_report(path)["schema"] == report["schema"]
    assert "engines" not in summary(report)


@pytest.mark.parametrize("field,value", [("target_unit", "minutes"), ("serving_enabled_for_agv", True),
                                         ("data_kind", "physical_robot")])
def test_registry_rejects_misleading_provenance(tmp_path, field, value):
    report = json.loads((ROOT / "docs/benchmarks/cmapss_fd001.json").read_text())
    report[field] = value
    path = tmp_path / "report.json"; path.write_text(json.dumps(report))
    with pytest.raises(ValueError): load_report(path)


def test_archive_checksum_rejection(tmp_path):
    archive = tmp_path / "changed.zip"; archive.write_bytes(b"tampered")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="checksum"):
        prepare_archive(archive, output)
    assert not output.exists()


def test_dataset_builder_preserves_asset_identity(tmp_path):
    from build_enhanced_dataset import main
    for split, asset in [("Training", "robot-A"), ("Validation", "robot-B")]:
        folder = tmp_path / split / "02.라벨링데이터"; folder.mkdir(parents=True)
        with zipfile.ZipFile(folder / "records.zip", "w") as archive:
            for i in range(35):
                archive.writestr(f"{i}.json", json.dumps({"deviceId": asset, "deviceType": "robot",
                    "createdAt": f"2026-01-01T00:00:{i:02d}", "errorData": {"errorCode": "정상"},
                    "deviceData": {"mainState": "MOVE", "batteryLevel": i}}))
    out = tmp_path / "built"
    main(["--base-dir", str(tmp_path), "--out-dir", str(out)])
    built = load_dataset(out / "enhanced_train.npz")
    assert built["features"].shape == (5, 249)
    assert set(built["groups"]) == {"robot-A"}
    with np.load(out / "enhanced_train.npz") as arrays:
        assert arrays["target_times"][0].endswith(":30")
        assert arrays["X"][0, -1, 0] == 29
    with zipfile.ZipFile(tmp_path / "Validation/02.라벨링데이터/records.zip", "a") as archive:
        archive.writestr("overlap.json", json.dumps({"deviceId": "robot-A", "deviceType": "robot",
            "errorData": {"errorCode": "정상"}, "deviceData": {"mainState": "MOVE"}}))
    with pytest.raises(ValueError, match="overlap"):
        main(["--base-dir", str(tmp_path), "--out-dir", str(out)])


def test_candidate_training_on_fixture_does_not_touch_champion(tmp_path):
    champion = ROOT / "data/processed"
    before = (champion / "robot_pdm_enhanced.txt").read_bytes()
    meta = json.loads((champion / "enhanced_meta.json").read_text())
    classes = len(meta["err_map"])
    rng = np.random.default_rng(7)
    for split, offset in [("train", 0), ("val", 100)]:
        groups = np.repeat(np.arange(12) + offset, classes * 10).astype(str)
        labels = np.tile(np.repeat(np.arange(classes), 10), 12)
        raw = rng.normal(size=(len(labels), 30, 7)).astype(np.float32)
        raw[:, :, 0] += labels[:, None] * 10
        np.savez(tmp_path / f"enhanced_{split}.npz", X=raw, S=np.zeros((len(labels), 9)), y=labels, groups=groups)
    (tmp_path / "enhanced_meta.json").write_text(json.dumps(meta))
    report = train_candidates(tmp_path, tmp_path / "candidate", champion, rounds=2)
    assert len(report["experiments"]) == 3
    assert not report["official_validation_used_for_selection"]
    assert not report["promotion"]["automatic_promotion"]
    assert not set(report["split_audit"]["fit_assets"]) & set(report["split_audit"]["selection_assets"])
    assert before == (champion / "robot_pdm_enhanced.txt").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        train_candidates(tmp_path, champion, champion)
    meta["devtype_map"] = {"incompatible": 999}
    (tmp_path / "enhanced_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="encoding"):
        train_candidates(tmp_path, tmp_path / "candidate", champion, rounds=2)
