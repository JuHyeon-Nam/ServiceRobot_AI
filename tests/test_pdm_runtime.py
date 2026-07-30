import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from pdm_runtime import load_runtime, predict_window, synthesize_live_window


def test_synthesize_live_window_contract():
    window, ctx = synthesize_live_window(
        seq_idx=10,
        n_frames=100,
        device_type="물류로봇",
        pred_hint="E-RBT-B",
        x=30.0,
        y=40.0,
        degree=90.0,
    )

    assert len(window) == 30
    assert all(len(row) == 7 for row in window)
    assert ctx["deviceType"] == "물류로봇"
    assert ctx["batteryUse"] > 50
    assert window[-1][0] < 25


def test_predict_window_uses_model_contract():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    booster, model_meta, dataset_meta = load_runtime(data_dir)
    window, ctx = synthesize_live_window(20, 100, "안내로봇", "정상", 10.0, 20.0, 0.0)

    result = predict_window(booster, model_meta, dataset_meta, window, ctx)

    assert result["pred"] in model_meta["class_names"]
    assert 0 <= result["conf"] <= 1
    assert result["n_features"] == model_meta["n_features"] == 249
    assert result["latency_ms"] >= 0
