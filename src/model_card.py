"""
model_card.py — model registry and responsible-AI metadata for the PdM model.

The serving layer should not only answer "what did the model predict?", but also
"which exact artifact is running, what inputs does it expect, how was it
validated, and when should operators distrust it?". This module builds that
contract from the checked-in model metadata and artifact bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from drift_monitor import DEFAULT_REFERENCE_PROFILE, DriftThresholds


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_model_card(data_dir: str | Path) -> dict[str, Any]:
    data = Path(data_dir)
    model_path = data / "robot_pdm_enhanced.txt"
    model_meta_path = data / "robot_pdm_enhanced_meta.json"
    dataset_meta_path = data / "enhanced_meta.json"
    model_meta = load_json(model_meta_path)
    dataset_meta = load_json(dataset_meta_path)

    dyn = list(model_meta["dyn"])
    used_idx = list(model_meta["model_dyn_idx"])
    used_dyn = [dyn[i] for i in used_idx]
    excluded_dyn = [name for i, name in enumerate(dyn) if i not in used_idx]
    stat = list(model_meta["stat"])
    raw_window_features = len(dyn) * 30
    engineered_features = len(used_dyn) * (30 + 1 + 1 + 1 + 15)
    expected_n_features = engineered_features + len(stat)

    return {
        "model_id": "robot-pdm-lightgbm-enhanced",
        "version": model_path.stem,
        "task": "9-class real-time service-robot fault diagnosis",
        "framework": "LightGBM native Booster",
        "artifact": {
            "path": "data/processed/robot_pdm_enhanced.txt",
            "format": "lightgbm_native_txt",
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },
        "metadata_files": {
            "model_meta": "data/processed/robot_pdm_enhanced_meta.json",
            "dataset_meta": "data/processed/enhanced_meta.json",
        },
        "performance": {
            "official_validation": {
                "accuracy": model_meta["val_acc"],
                "macro_f1": model_meta["val_macro_f1"],
                "split": "AI-Hub official validation split; robots unseen during training",
            },
            "best_iteration": model_meta["best_iteration"],
            "baseline": {
                "strategy": "always predict 정상",
                "accuracy": 0.83,
            },
        },
        "input_contract": {
            "window_timesteps": 30,
            "raw_dynamic_sensors": dyn,
            "model_dynamic_sensors": used_dyn,
            "excluded_dynamic_sensors": excluded_dyn,
            "static_context": stat,
            "categorical_maps": {
                "deviceType": dataset_meta.get("devtype_map", {}),
                "mainState": dataset_meta.get("mainstate_map", {}),
                "crowd": dataset_meta.get("crowd_map", {}),
            },
        },
        "feature_engineering": {
            "steps": [
                "drop absolute x/y coordinates for cross-site generalization",
                "flatten 30-step dynamic sequence",
                "append mean, standard deviation, and early-vs-late drift features",
                "append first 15 rFFT magnitudes per retained dynamic sensor",
                "append static and context features",
            ],
            "raw_window_features": raw_window_features,
            "engineered_dynamic_features": engineered_features,
            "static_features": len(stat),
            "n_features": model_meta["n_features"],
            "expected_n_features": expected_n_features,
            "contract_ok": expected_n_features == model_meta["n_features"],
        },
        "labels": {
            "classes": model_meta["class_names"],
            "err_map": model_meta["err_map"],
        },
        "data": {
            "source": "AI-Hub indoor-space maintenance service robot dataset",
            "raw_scale": "4.2GB JSON dataset; processed artifacts are checked in for demo reproducibility",
            "known_limitations": [
                "rare classes such as E-RBT-N and E-RBT-S have very small validation support",
                "some faults overlap normal sensor ranges, so missed detections must be monitored",
                "the demo stream uses replay trajectories and deterministic synthetic sensor windows rather than a physical robot feed",
            ],
        },
        "operations": {
            "serving_endpoints": ["/predict", "/health", "/model-card", "/api/model-card"],
            "observability": ["/metrics", "/api/data-quality", "/api/drift", "/api/reliability"],
            "explainability": "LightGBM contribution-based top physical signal groups returned by /predict",
            "live_inference": "The 3D twin /api/snapshot path runs the LightGBM Booster on each uncached AGV frame; replay prediction is retained only as an audit hint.",
            "drift_monitoring": {
                "reference_profile": DEFAULT_REFERENCE_PROFILE,
                "thresholds": DriftThresholds().__dict__,
            },
            "retrain_triggers": [
                "/api/drift status remains drift for several monitoring windows",
                "QA pass rate falls below the release threshold",
                "new robot type, site layout, or sensor calibration profile is introduced",
                "post-maintenance labels show rising missed-detection incidents",
            ],
        },
    }
