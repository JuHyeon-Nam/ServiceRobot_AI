"""
pdm_runtime.py — shared PdM model runtime utilities.

Both the standalone inference API and the real-time digital twin use this module
so feature construction, model loading, and live prediction stay on one contract.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


MODEL_DYN_IDX = [0, 1, 4, 5, 6]  # batteryLevel, speed, degree, collision, obstacle


def load_booster(path: str | Path) -> lgb.Booster:
    """Load a LightGBM native text model safely, including non-ASCII paths."""
    with open(path, "r", encoding="utf-8") as f:
        return lgb.Booster(model_str=f.read())


def load_runtime(data_dir: str | Path) -> tuple[lgb.Booster, dict[str, Any], dict[str, Any]]:
    data = Path(data_dir)
    booster = load_booster(data / "robot_pdm_enhanced.txt")
    with open(data / "robot_pdm_enhanced_meta.json", encoding="utf-8") as f:
        model_meta = json.load(f)
    with open(data / "enhanced_meta.json", encoding="utf-8") as f:
        dataset_meta = json.load(f)
    return booster, model_meta, dataset_meta


def _get(ctx: Any, key: str, default: Any = 0) -> Any:
    if isinstance(ctx, dict):
        return ctx.get(key, default)
    return getattr(ctx, key, default)


def make_features(window: list, ctx: Any, dataset_meta: dict[str, Any]) -> np.ndarray:
    """Build the exact 249-feature vector used by training and serving."""
    devmap = dataset_meta["devtype_map"]
    mainmap = dataset_meta.get("mainstate_map", {})
    crowdmap = dataset_meta.get("crowd_map", {"LOW": 0, "MIDDLE": 1, "HIGH": 2})

    xf = np.asarray(window, dtype=np.float32).reshape(1, 30, 7)
    x = xf[:, :, MODEL_DYN_IDX]
    eng = np.hstack([
        x.reshape(1, -1),
        np.mean(x, 1),
        np.std(x, 1),
        np.mean(x[:, -10:, :], 1) - np.mean(x[:, :10, :], 1),
        np.abs(np.fft.rfft(x, axis=1))[:, :15, :].reshape(1, -1),
    ])
    stat = np.array([[
        _get(ctx, "isOffline", 0),
        _get(ctx, "nowCharging", 0),
        _get(ctx, "emergencyStop", 0),
        _get(ctx, "batteryUse", 0),
        _get(ctx, "batteryCycleCount", 0),
        _get(ctx, "distance", 0),
        crowdmap.get(_get(ctx, "crowd", "MIDDLE"), 1),
        devmap.get(_get(ctx, "deviceType", "안내로봇"), 0),
        mainmap.get(_get(ctx, "mainState", "MOVE"), 0),
    ]], dtype=np.float32)
    return np.hstack([eng, stat]).astype(np.float32)


def predict_window(
    booster: lgb.Booster,
    model_meta: dict[str, Any],
    dataset_meta: dict[str, Any],
    window: list,
    context: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    feats = make_features(window, context, dataset_meta)
    probs = booster.predict(feats)[0]
    top = int(np.argmax(probs))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "pred": model_meta["class_names"][top],
        "conf": float(probs[top]),
        "latency_ms": round(elapsed_ms, 4),
        "n_features": int(feats.shape[1]),
    }


def synthesize_live_window(
    seq_idx: int,
    n_frames: int,
    device_type: str,
    pred_hint: str,
    x: float,
    y: float,
    degree: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Create a deterministic 30-step sensor window for live twin inference.

    The route/AGV motion still comes from replay trajectories, but the diagnosis
    now flows through the LightGBM Booster on every uncached frame. pred_hint is
    used only to shape a plausible operating scenario for the simulator.
    """
    phase = seq_idx / max(n_frames - 1, 1)
    base_batt = max(8.0, 92.0 - 42.0 * phase)
    base_speed = 0.65 + 0.18 * math.sin(seq_idx * 0.17)
    collision = 0.0
    obstacle = 0.0
    is_offline = 0.0
    charging = 0.0
    estop = 0.0
    crowd = "MIDDLE"
    main_state = "MOVE"
    battery_use = 12.0 + 42.0 * phase
    cycle_count = 40.0 + 260.0 * phase
    distance = 8000.0 + 82000.0 * phase

    if pred_hint == "E-RBT-B":
        base_batt = 16.0 + 2.0 * math.sin(seq_idx * 0.11)
        battery_use += 55
        cycle_count += 420
        main_state = "MOVE"
    elif pred_hint == "E-RBT-E":
        base_speed = 0.04
        estop = 1.0
        main_state = "ESTOP"
    elif pred_hint == "E-RBT-N":
        is_offline = 1.0
        main_state = "OFFLINE"
    elif pred_hint == "E-RBT-S":
        base_speed = 0.45
        main_state = "ERROR"
    elif pred_hint == "E-ENV-C":
        collision = 1.0
        crowd = "HIGH"
        base_speed = 0.22
    elif pred_hint == "E-ENV-O":
        obstacle = 1.0
        crowd = "HIGH"
        base_speed = 0.18
    elif pred_hint == "E-INF-A":
        base_speed = 0.05
        main_state = "PAUSED"
    elif pred_hint == "E-INF-E":
        base_speed = 0.06
        main_state = "PAUSED"
        charging = 1.0
    elif base_batt < 25:
        charging = 1.0
        main_state = "CHARGING"

    window = []
    for t in range(30):
        k = t - 29
        wobble = math.sin((seq_idx + t) * 0.31)
        speed = max(0.0, base_speed + 0.04 * wobble)
        deg = degree + 2.5 * math.sin((seq_idx + t) * 0.19)
        if pred_hint == "E-RBT-S":
            deg += 22.0 * math.sin((seq_idx + t) * 1.7)
            speed += 0.35 * abs(math.sin((seq_idx + t) * 1.3))
        batt = max(3.0, min(100.0, base_batt - 0.08 * (29 - t)))
        window.append([
            batt,
            speed,
            x + k * speed * 0.15,
            y + k * speed * 0.12,
            deg,
            collision,
            obstacle,
        ])

    context = {
        "deviceType": device_type,
        "mainState": main_state,
        "crowd": crowd,
        "isOffline": is_offline,
        "nowCharging": charging,
        "emergencyStop": estop,
        "batteryUse": battery_use,
        "batteryCycleCount": cycle_count,
        "distance": distance,
    }
    return window, context
