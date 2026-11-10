"""Synthetic equivalence and batch-size bounds, not real-data RAM/performance claims."""
import gc
import hashlib
from pathlib import Path
import sys
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pdm_runtime
import train_diagnosis
from train_diagnosis import load_dataset


@pytest.mark.parametrize("rows,batch", [(17, 1), (17, 3), (4096, 4096), (4099, 4096), (17, 100), (0, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_features_exactly_match_existing_formula(rows, batch, dtype):
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(rows, 30, 7)).astype(dtype)
    static = rng.normal(size=(rows, 9)).astype(dtype)
    expected = pdm_runtime.make_feature_matrix(raw, static)
    actual = pdm_runtime.make_feature_matrix_batched(raw, static, batch)
    assert actual.dtype == np.float32
    assert actual.shape == (rows, 249)
    np.testing.assert_array_equal(actual, expected)


def test_default_bounds_temporaries_and_passes_views_before_dtype_conversion(monkeypatch):
    rows = 2 * pdm_runtime.FEATURE_BATCH_ROWS + 1
    rng = np.random.default_rng(7)
    raw, static = rng.normal(size=(rows, 30, 7)), rng.normal(size=(rows, 9))
    original = pdm_runtime.make_feature_matrix
    expected = original(raw, static)
    calls = []

    def tracked(windows, contexts):
        assert windows.dtype == contexts.dtype == np.float64
        assert np.shares_memory(windows, raw) and np.shares_memory(contexts, static)
        calls.append(len(windows))
        return original(windows, contexts)

    monkeypatch.setattr(pdm_runtime, "make_feature_matrix", tracked)
    actual = pdm_runtime.make_feature_matrix_batched(raw, static)
    assert calls == [4096, 4096, 1]
    np.testing.assert_array_equal(actual, expected)


def test_noncontiguous_inputs_are_not_modified():
    rng = np.random.default_rng(13)
    raw = rng.normal(size=(38, 30, 7))[::2]
    static = rng.normal(size=(38, 18))[::2, ::2]
    assert not raw.flags.c_contiguous and not static.flags.c_contiguous
    before_raw, before_static = raw.copy(), static.copy()
    actual = pdm_runtime.make_feature_matrix_batched(raw, static, 4)
    np.testing.assert_array_equal(actual, pdm_runtime.make_feature_matrix(raw, static))
    np.testing.assert_array_equal(raw, before_raw)
    np.testing.assert_array_equal(static, before_static)


@pytest.mark.parametrize("batch", [0, -1, True, np.bool_(True), 1.5, "8", None, float("inf")])
def test_invalid_batch_sizes_are_rejected(batch):
    with pytest.raises(ValueError, match="feature batch size must be a positive integer"):
        pdm_runtime.make_feature_matrix_batched(np.zeros((1, 30, 7)), np.zeros((1, 9)), batch)


@pytest.mark.parametrize("raw_shape,static_shape,message", [
    ((1, 29, 7), (1, 9), "sensor windows"),
    ((1, 30, 6), (1, 9), "sensor windows"),
    ((30, 7), (1, 9), "sensor windows"),
    ((1, 30, 7), (2, 9), "context features"),
    ((1, 30, 7), (1, 8), "context features"),
])
def test_shape_errors_stop_before_feature_computation(monkeypatch, raw_shape, static_shape, message):
    def forbidden(*args):
        pytest.fail("invalid shape must be rejected before computing any feature batch")

    monkeypatch.setattr(pdm_runtime, "make_feature_matrix", forbidden)
    with pytest.raises(ValueError, match=message):
        pdm_runtime.make_feature_matrix_batched(np.zeros(raw_shape), np.zeros(static_shape), 4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["raw", "static"])
def test_nonfinite_last_batch_is_not_returned_as_valid(monkeypatch, value, field):
    raw, static = np.zeros((9, 30, 7)), np.zeros((9, 9))
    if field == "raw":
        raw[-1, -1, -1] = value
    else:
        static[-1, -1] = value
    original = pdm_runtime.make_feature_matrix
    calls = []

    def tracked(windows, contexts):
        calls.append(len(windows))
        return original(windows, contexts)

    monkeypatch.setattr(pdm_runtime, "make_feature_matrix", tracked)
    with pytest.raises(ValueError, match="must be finite"):
        pdm_runtime.make_feature_matrix_batched(raw, static, 4)
    assert calls == [4, 4, 1]


def test_dataset_loader_uses_batches_and_preserves_npz_alignment(tmp_path, monkeypatch):
    rows = pdm_runtime.FEATURE_BATCH_ROWS + 1
    rng = np.random.default_rng(21)
    raw, static = rng.normal(size=(rows, 30, 7)), rng.normal(size=(rows, 9))
    labels, groups = np.arange(rows) % 9, np.array([f"robot-{i % 4}" for i in range(rows)])
    path = tmp_path / "synthetic-windows.npz"
    np.savez_compressed(path, X=raw, S=static, y=labels, groups=groups)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    original = pdm_runtime.make_feature_matrix
    expected = original(raw, static)
    calls = []

    def tracked(windows, contexts):
        calls.append(len(windows))
        return original(windows, contexts)

    monkeypatch.setattr(pdm_runtime, "make_feature_matrix", tracked)
    loaded = load_dataset(path)
    assert calls == [4096, 1]
    np.testing.assert_array_equal(loaded["features"], expected)
    np.testing.assert_array_equal(loaded["X"], raw)
    np.testing.assert_array_equal(loaded["S"], static)
    np.testing.assert_array_equal(loaded["y"], labels)
    np.testing.assert_array_equal(loaded["groups"], groups)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.fixture
def retention_windows(tmp_path):
    rng = np.random.default_rng(31)
    path = tmp_path / "retention-fixture.npz"
    np.savez_compressed(path, X=rng.normal(size=(17, 30, 7)).astype(np.float32),
                        S=rng.normal(size=(17, 9)).astype(np.float32), y=np.arange(17) % 9,
                        groups=np.array([f"robot-{i % 4}" for i in range(17)]))
    return path


@pytest.mark.parametrize("with_features", [True, False])
def test_lean_loading_preserves_labels_groups_features_and_file(retention_windows, with_features):
    before = hashlib.sha256(retention_windows.read_bytes()).hexdigest()
    full = load_dataset(retention_windows, with_features=with_features)
    lean = load_dataset(retention_windows, with_features=with_features, keep_raw=False)
    assert {"X", "S"} <= full.keys()
    assert set(lean) == set(full) - {"X", "S"}
    for key in lean:
        np.testing.assert_array_equal(lean[key], full[key])
    assert ("features" in lean) == with_features
    assert hashlib.sha256(retention_windows.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("keep_raw", [True, False])
def test_loaded_sensor_arrays_can_be_collected_when_not_retained(retention_windows, monkeypatch, keep_raw):
    original = train_diagnosis.make_feature_matrix_batched
    refs = []

    def tracked(raw, static):
        refs.extend([weakref.ref(raw), weakref.ref(static)])
        return original(raw, static)

    monkeypatch.setattr(train_diagnosis, "make_feature_matrix_batched", tracked)
    result = load_dataset(retention_windows, keep_raw=keep_raw)
    gc.collect()
    assert len(refs) == 2
    if keep_raw:
        assert refs[0]() is result["X"] and refs[1]() is result["S"]
    else:
        assert all(ref() is None for ref in refs)
        assert result["features"].shape == (17, 249)


@pytest.mark.parametrize("with_features", [True, False])
@pytest.mark.parametrize("kind,message", [("sensor_nan", "must be finite"), ("context_inf", "must be finite"),
                                           ("missing_asset", "asset IDs"), ("float_label", "class labels")])
def test_lean_loading_does_not_skip_raw_input_validation(retention_windows, monkeypatch, with_features, kind, message):
    with np.load(retention_windows, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if kind == "sensor_nan":
        arrays["X"][-1, -1, -1] = np.nan
    elif kind == "context_inf":
        arrays["S"][-1, -1] = np.inf
    elif kind == "missing_asset":
        arrays["groups"] = np.full(17, "null")
    else:
        arrays["y"] = arrays["y"].astype(float)
    np.savez_compressed(retention_windows, **arrays)

    def forbidden(*args):
        pytest.fail("invalid raw input must be rejected before feature computation")

    monkeypatch.setattr(train_diagnosis, "make_feature_matrix_batched", forbidden)
    with pytest.raises(ValueError, match=message):
        load_dataset(retention_windows, with_features=with_features, keep_raw=False)
