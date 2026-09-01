"""
train_rul_baseline.py — train a calibrated RUL regression baseline.

This trainer consumes the supervised-ready CSV produced by build_rul_dataset.py.
Only rows with observed future failures are used for regression; right-censored
rows remain part of the dataset contract but require a survival model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from rul_runtime import RUL_FEATURES, RUL_SCHEMA, RUL_TARGET


MODEL_SCHEMA = "fab.rul.baseline_model.v1"


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rul_training_csv(path: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("event_observed")) != "1" or row.get(RUL_TARGET) in (None, ""):
                continue
            rows.append(row)
    if len(rows) < 4:
        raise ValueError("RUL baseline needs at least 4 observed failure rows")
    x = np.asarray([[_num(row.get(name)) for name in RUL_FEATURES] for row in rows], dtype=np.float32)
    y = np.asarray([_num(row.get(RUL_TARGET)) for row in rows], dtype=np.float32)
    return x, y, rows


def train_rul_baseline(path: str, random_state: int = 42, test_size: float = 0.3) -> dict:
    t0 = time.time()
    x, y, rows = load_rul_training_csv(path)
    if len(rows) < 8:
        x_train, x_test, y_train, y_test = x, x, y, y
        split = "train_eval_same_small_sample"
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        split = "random_holdout"

    model = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        random_state=random_state,
    )
    baseline = DummyRegressor(strategy="median")
    model.fit(x_train, y_train)
    baseline.fit(x_train, y_train)

    pred = model.predict(x_test)
    base_pred = baseline.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    base_mae = mean_absolute_error(y_test, base_pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    base_rmse = mean_squared_error(y_test, base_pred) ** 0.5
    improvement = 0.0 if base_mae == 0 else (base_mae - mae) / base_mae

    return {
        "schema": MODEL_SCHEMA,
        "contract_schema": RUL_SCHEMA,
        "target": RUL_TARGET,
        "model_kind": "gradient_boosting_regressor",
        "runtime_mode": "offline_calibrated_baseline",
        "feature_fields": list(RUL_FEATURES),
        "rows": len(rows),
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "split": split,
        "metrics": {
            "mae_min": round(float(mae), 4),
            "rmse_min": round(float(rmse), 4),
            "r2": round(float(r2_score(y_test, pred)), 4) if len(y_test) > 1 else None,
            "median_baseline_mae_min": round(float(base_mae), 4),
            "median_baseline_rmse_min": round(float(base_rmse), 4),
            "mae_improvement_vs_median": round(float(improvement), 4),
        },
        "model": model,
        "trained_at_epoch": round(time.time(), 3),
        "elapsed_sec": round(time.time() - t0, 3),
        "limitations": [
            "Uses observed failure rows only; censored rows require survival modeling.",
            "Small synthetic smoke fixtures prove the pipeline, not field accuracy.",
        ],
    }


def save_artifacts(result: dict, model_path: str, meta_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "schema": result["schema"],
            "feature_fields": result["feature_fields"],
            "target": result["target"],
            "model": result["model"],
        }, f)
    metadata = {k: v for k, v in result.items() if k != "model"}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train an offline RUL regression baseline from rul_training.csv.")
    p.add_argument("--train-csv", default="../data/processed/rul_training.csv")
    p.add_argument("--model-out", default="../data/processed/rul_baseline.pkl")
    p.add_argument("--meta-json", default="../data/processed/rul_baseline_meta.json")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args(argv)

    result = train_rul_baseline(args.train_csv, random_state=args.random_state)
    metadata = save_artifacts(result, args.model_out, args.meta_json)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
