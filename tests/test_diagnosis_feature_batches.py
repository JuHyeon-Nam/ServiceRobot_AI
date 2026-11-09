"""Synthetic equivalence and batch-size bounds, not real-data RAM/performance claims."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pdm_runtime
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
