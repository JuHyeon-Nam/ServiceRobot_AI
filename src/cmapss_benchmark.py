"""Engine-level, causal RUL benchmark, independent of the AGV runtime model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["unit", "cycle", *[f"setting_{i}" for i in range(1, 4)],
           *[f"sensor_{i}" for i in range(1, 22)]]
SIGNALS = COLUMNS[2:]
SCHEMA = "servicerobot.cmapss.benchmark.v1"


def read_trajectories(path: Path) -> pd.DataFrame:
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape[1] != 26 or not np.isfinite(raw).all():
        raise ValueError("C-MAPSS requires 26 finite numeric columns")
    if (raw[:, :2] < 1).any() or (raw[:, :2] != np.floor(raw[:, :2])).any():
        raise ValueError("unit and cycle must be positive integers")
    frame = pd.DataFrame(raw, columns=COLUMNS)
    frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(int)
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError("duplicate unit/cycle")
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    for _, group in frame.groupby("unit"):
        if not np.array_equal(group.cycle.to_numpy(), np.arange(1, len(group) + 1)):
            raise ValueError("each trajectory must have consecutive cycles from 1")
    return frame


def causal_features(frame: pd.DataFrame, temporal: bool, lookback: int = 20) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    parts = []
    for _, group in frame.groupby("unit", sort=False):
        signals = group[SIGNALS]
        features = signals.copy()
        features["cycle"] = group.cycle
        if temporal:
            rolling = signals.rolling(lookback, min_periods=1)
            features = pd.concat([
                features, rolling.mean().add_suffix("_mean"),
                rolling.std(ddof=0).fillna(0).add_suffix("_std"),
                (signals - signals.shift(lookback - 1).fillna(signals.iloc[0])).add_suffix("_delta"),
            ], axis=1)
        parts.append(features)
    return pd.concat(parts).sort_index().astype(np.float32)


def split_units(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    units = frame.unit.to_numpy()
    if len(np.unique(units)) < 5:
        raise ValueError("benchmark needs at least 5 independent training engines")
    fit, held = next(GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed).split(frame, groups=units))
    selection_local, calibration_local = next(GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=seed + 1
    ).split(frame.iloc[held], groups=units[held]))
    return fit, held[selection_local], held[calibration_local]


def truncated_indices(frame: pd.DataFrame, rows: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = []
    for _, group in frame.iloc[rows].groupby("unit", sort=True):
        cut = max(1, int(math.floor(len(group) * rng.uniform(0.4, 0.85))))
        indices.append(group.index[cut - 1])
    return np.asarray(indices, dtype=int)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "mae_cycles": round(float(np.mean(np.abs(error))), 4),
        "rmse_cycles": round(float(np.sqrt(np.mean(error ** 2))), 4),
        "late_prediction_rate": round(float(np.mean(error > 0)), 4),
        "p95_absolute_error_cycles": round(float(np.quantile(np.abs(error), 0.95)), 4),
    }


def interval_radius(residuals: np.ndarray, coverage: float) -> float:
    residuals = np.asarray(residuals, dtype=float)
    if not 0 < coverage < 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("invalid conformal calibration inputs")
    rank = math.ceil((len(residuals) + 1) * coverage)
    if rank > len(residuals):
        raise ValueError("more calibration engines needed for finite conformal interval")
    return float(np.sort(np.abs(residuals))[rank - 1])


def fit_model(features: pd.DataFrame, labels: np.ndarray, frame: pd.DataFrame,
              rows: np.ndarray, leaves: int, seed: int, rounds: int) -> tuple[lgb.LGBMRegressor, list[str]]:
    usable = features.iloc[rows].var().gt(1e-10)
    columns = features.columns[usable].tolist()
    lengths = frame.iloc[rows].groupby("unit").size()
    weights = 1 / frame.iloc[rows].unit.map(lengths).to_numpy()
    weights /= weights.mean()
    model = lgb.LGBMRegressor(
        n_estimators=rounds, learning_rate=0.04, num_leaves=leaves,
        max_depth=-1, min_child_samples=35, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=seed, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    model.fit(features.iloc[rows][columns], labels[rows], sample_weight=weights)
    return model, columns


def run_benchmark(data: Path, output: Path, report_path: Path,
                  seed: int = 42, lookback: int = 20, cap: float = 125,
                  rounds: int = 350, coverage: float = 0.9) -> dict:
    if not np.isfinite(cap) or cap <= 0 or rounds < 1:
        raise ValueError("cap and rounds must be positive")
    train = read_trajectories(data / "train_FD001.txt")
    fit, selection, calibration = split_units(train, seed)
    lifetime = train.groupby("unit").cycle.transform("max").to_numpy()
    true_train = lifetime - train.cycle.to_numpy()
    target = np.minimum(true_train, cap)
    select_points = truncated_indices(train, selection, seed + 2)
    calibration_points = truncated_indices(train, calibration, seed + 3)
    feature_sets = {name: causal_features(train, name == "temporal", lookback)
                    for name in ("snapshot", "temporal")}
    candidates = []
    for name, features in feature_sets.items():
        for leaves in (15, 31):
            model, columns = fit_model(features, target, train, fit, leaves, seed, rounds)
            pred = np.clip(model.predict(features.iloc[select_points][columns]), 0, cap)
            candidates.append({"features": name, "leaves": leaves,
                               "selection": regression_metrics(true_train[select_points], pred)})
    winner = min(candidates, key=lambda row: row["selection"]["rmse_cycles"])
    refit = np.sort(np.concatenate([fit, selection]))
    model, columns = fit_model(feature_sets[winner["features"]], target, train, refit,
                               winner["leaves"], seed, rounds)
    calibrated = np.clip(model.predict(feature_sets[winner["features"]].iloc[calibration_points][columns]), 0, cap)
    radius = interval_radius(true_train[calibration_points] - calibrated, coverage)

    # Official test trajectories and labels are read only after model selection.
    test = read_trajectories(data / "test_FD001.txt")
    truth = np.loadtxt(data / "RUL_FD001.txt", ndmin=1)
    units = sorted(test.unit.unique())
    if truth.ndim != 1 or len(truth) != len(units) or not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("test RUL labels must match engines and be finite/nonnegative")
    if units != list(range(1, len(units) + 1)):
        raise ValueError("test unit IDs must match the RUL file's 1-based row order")
    test_features = causal_features(test, winner["features"] == "temporal", lookback)
    predictions = np.clip(model.predict(test_features[columns]), 0, cap)
    last = test.groupby("unit", sort=True).tail(1)
    last_pred = predictions[last.index]
    lower, upper = np.maximum(0, last_pred - radius), last_pred + radius
    baseline_value = float(np.median(target[refit]))
    baseline = regression_metrics(truth, np.full(len(truth), baseline_value))
    measured = regression_metrics(truth, last_pred)
    measured["interval_coverage"] = round(float(np.mean((truth >= lower) & (truth <= upper))), 4)
    measured["interval_mean_width_cycles"] = round(float(np.mean(upper - lower)), 4)
    truth_by_unit = dict(zip(units, truth))
    engines = []
    for unit, group in test.groupby("unit", sort=True):
        final = group.iloc[-1]
        history = []
        for index, row in group.tail(30).iterrows():
            actual = float(truth_by_unit[unit] + final.cycle - row.cycle)
            estimate = float(predictions[index])
            history.append({"cycle": int(row.cycle), "true_rul": actual,
                            "prediction": round(estimate, 3),
                            "lower": round(max(0, estimate - radius), 3),
                            "upper": round(estimate + radius, 3)})
        engines.append({"unit": int(unit), "last_cycle": int(final.cycle),
                        "true_rul": float(truth_by_unit[unit]),
                        "prediction": round(float(predictions[final.name]), 3),
                        "history": history})
    files = {}
    for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        files[name] = hashlib.sha256((data / name).read_bytes()).hexdigest()
    source = json.loads((data / "manifest.json").read_text(encoding="utf-8")) if (data / "manifest.json").exists() else {"verification": "local_files_only"}
    report = {
        "schema": SCHEMA, "dataset": "NASA C-MAPSS FD001",
        "data_kind": "engine_degradation_simulation", "target_unit": "cycles",
        "source": source, "file_sha256": files,
        "protocol": {"seed": seed, "lookback": lookback, "training_target_cap": cap,
                     "test_target_capped": False, "test_points": "last_observed_cycle_per_engine",
                     "feature_timing": "causal_past_and_current_only",
                     "selection": "training_engine_holdout_only",
                     "train_test_unit_namespace": "separate_engine_cohorts",
                     "fit_units": sorted(map(int, train.iloc[fit].unit.unique())),
                     "selection_units": sorted(map(int, train.iloc[selection].unit.unique())),
                     "calibration_units": sorted(map(int, train.iloc[calibration].unit.unique())),
                     "asset_overlap": 0, "refit_engines": int(train.iloc[refit].unit.nunique()),
                     "calibration_engines": len(calibration_points),
                     "test_engines": len(units), "train_rows": len(train), "test_rows": len(test)},
        "candidates": candidates, "selected": winner, "feature_count": len(columns),
        "baseline": {"kind": "training_target_median", "prediction": baseline_value, **baseline},
        "test_metrics": measured,
        "capped_test_metrics": regression_metrics(np.minimum(truth, cap), last_pred),
        "mae_improvement_vs_baseline": round(1 - measured["mae_cycles"] / baseline["mae_cycles"], 4) if baseline["mae_cycles"] else None,
        "uncertainty": {"method": "split_conformal_absolute_residual", "nominal_coverage": coverage,
                        "radius_cycles": round(radius, 4), "calibration_points": len(calibration_points)},
        "engines": engines, "serving_enabled_for_agv": False,
        "limitations": ["NASA simulator data, not physical robot field measurements.",
                         "Engine operating cycles cannot be converted to AGV minutes.",
                         "One FD001 test evaluation; not cross-condition or field validation.",
                         "Conformal intervals require exchangeability; coverage is not guaranteed under domain shift.",
                         "History intervals use endpoint calibration; per-cycle coverage was not calibrated."],
    }
    output.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output / "rul_model.txt"))
    (output / "rul_model_meta.json").write_text(json.dumps({
        "schema": SCHEMA, "features": columns, "lookback": lookback, "cap": cap,
        "selected": winner, "interval_radius": radius, "agv_serving": False,
    }, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/external/cmapss")
    parser.add_argument("--output", type=Path, default=ROOT / "data/experiments/cmapss-fd001")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/benchmarks/cmapss_fd001.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=350)
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_benchmark(args.data_dir, args.output, args.report, args.seed, args.lookback,
                           rounds=args.rounds)
    print(json.dumps({"selected": report["selected"], "test_metrics": report["test_metrics"],
                      "baseline": report["baseline"], "elapsed_sec": round(time.perf_counter() - started, 2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
