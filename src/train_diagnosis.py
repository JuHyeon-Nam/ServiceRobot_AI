"""Select diagnosis candidates on robot holdout; evaluate once on official validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from diagnosis_evaluation import asset_holdout, class_weights, classification_metrics, promotion_gate
from pdm_runtime import MODEL_DYN_IDX, load_booster, make_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
NORMAL_RATIOS = (None, 2, 3, 5)
WEIGHT_MODES = ("unweighted", "sqrt_balanced", "balanced")
DATA_KINDS = ("unspecified", "aihub_recorded", "synthetic_fixture")


def load_dataset(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if not {"X", "S", "y", "groups"} <= set(data.files):
            raise ValueError("dataset needs X/S/y/groups; rebuild with build_enhanced_dataset.py")
        rows = {key: data[key] for key in ("X", "S", "y", "groups")}
    count = len(rows["y"])
    if not count or rows["y"].shape != (count,) or rows["groups"].shape != (count,):
        raise ValueError("invalid dataset row alignment")
    if not np.issubdtype(rows["y"].dtype, np.integer):
        raise ValueError("class labels must be integer indices")
    rows["features"] = make_feature_matrix(rows["X"], rows["S"])
    if len(rows["features"]) != count or any(not str(g).strip() for g in rows["groups"]):
        raise ValueError("invalid features or asset IDs")
    rows["groups"] = rows["groups"].astype(str)
    return rows


def training_rows(labels: np.ndarray, indices: np.ndarray, normal: int, seed: int,
                  normal_ratio: int | None = 3) -> np.ndarray:
    if normal_ratio is None:
        return np.sort(indices.copy())
    if isinstance(normal_ratio, bool) or not isinstance(normal_ratio, int) or normal_ratio < 1:
        raise ValueError("normal ratio must be a positive integer or None for all rows")
    rng = np.random.default_rng(seed)
    faults = indices[labels[indices] != normal]
    normals = indices[labels[indices] == normal]
    keep = rng.choice(normals, min(len(normals), len(faults) * normal_ratio), replace=False)
    return np.sort(np.concatenate([faults, keep]))


def dataset_support(labels: np.ndarray, groups: np.ndarray, names: list[str]) -> dict:
    return {name: {"rows": int(np.sum(labels == i)), "assets": int(len(np.unique(groups[labels == i])))}
            for i, name in enumerate(names)}


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_event(output: Path, event: str, **fields) -> None:
    row = {"event": event, "time_utc": datetime.now(timezone.utc).isoformat(), **fields}
    with (output / "training_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def train_candidates(data_dir: Path, output: Path, champion_dir: Path,
                     seed: int = 42, rounds: int = 800, max_false_alarm: float = 0.05,
                     normal_ratios: tuple[int | None, ...] = NORMAL_RATIOS,
                     weight_modes: tuple[str, ...] = WEIGHT_MODES, data_kind: str = "unspecified") -> dict:
    destination, serving = output.resolve(), champion_dir.resolve()
    if destination == serving or destination in serving.parents or serving in destination.parents:
        raise ValueError("candidate output must not overwrite the serving model directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("candidate output must be empty; use a new directory for each experiment")
    if rounds < 1 or not 0 <= max_false_alarm <= 1 or data_kind not in DATA_KINDS:
        raise ValueError("invalid rounds, false-alarm limit, or data kind")
    if (not normal_ratios or len(set(normal_ratios)) != len(normal_ratios)
            or any(r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 1) for r in normal_ratios)):
        raise ValueError("normal ratios must be unique positive integers or None")
    if (not weight_modes or len(set(weight_modes)) != len(weight_modes)
            or any(m not in WEIGHT_MODES for m in weight_modes)):
        raise ValueError("weight modes must be unique supported values")
    train, validation = load_dataset(data_dir / "enhanced_train.npz"), load_dataset(data_dir / "enhanced_val.npz")
    overlap = sorted(set(train["groups"]) & set(validation["groups"]))
    if overlap:
        raise ValueError(f"official train/validation asset overlap: {overlap[:5]}")
    metadata = json.loads((data_dir / "enhanced_meta.json").read_text(encoding="utf-8"))
    champion_dataset_meta = json.loads((champion_dir / "enhanced_meta.json").read_text(encoding="utf-8"))
    for key in ("devtype_map", "mainstate_map", "crowd_map", "dyn", "stat"):
        if metadata[key] != champion_dataset_meta[key]:
            raise ValueError(f"champion and candidate encoding contracts differ: {key}")
    ordered = sorted(metadata["err_map"].items(), key=lambda pair: pair[1])
    if [value for _, value in ordered] != list(range(len(ordered))):
        raise ValueError("class mapping must use contiguous zero-based indices")
    names = [name for name, _ in ordered]
    if set(train["y"].tolist()) != set(range(len(names))):
        raise ValueError("training dataset must contain all declared diagnosis classes")
    if (validation["y"] < 0).any() or (validation["y"] >= len(names)).any():
        raise ValueError("unknown validation class")
    champion_meta = json.loads((champion_dir / "robot_pdm_enhanced_meta.json").read_text(encoding="utf-8"))
    if champion_meta["class_names"] != names or champion_meta["n_features"] != 249:
        raise ValueError("champion and candidate class/feature contracts differ")
    normal = names.index("정상")
    fit, selection = asset_holdout(train["y"], train["groups"], seed, require_all_classes=True)
    train_features = pd.DataFrame(train["features"])
    validation_features = pd.DataFrame(validation["features"])
    lineage = {"data_kind": data_kind, "data_kind_source": "caller_declared_not_independently_verified",
               "sha256": {name: file_sha256(data_dir / name)
                          for name in ("enhanced_train.npz", "enhanced_val.npz", "enhanced_meta.json")},
               "code_sha256": {name: file_sha256(ROOT / "src" / name)
                               for name in ("train_diagnosis.py", "diagnosis_evaluation.py", "pdm_runtime.py")},
               "champion_sha256": file_sha256(champion_dir / "robot_pdm_enhanced.txt")}
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                   "lightgbm": lgb.__version__, "scikit_learn": sklearn.__version__}
    output.mkdir(parents=True, exist_ok=True)
    record_event(output, "run_started", seed=seed, rounds=rounds, lineage=lineage,
                 normal_ratios=list(normal_ratios), weight_modes=list(weight_modes), environment=environment)
    experiments = []
    for ratio in normal_ratios:
        indices = training_rows(train["y"], fit, normal, seed, ratio)
        for mode in weight_modes:
            candidate_id = f"normal_{'all' if ratio is None else str(ratio) + 'x'}__{mode}"
            record_event(output, "candidate_started", candidate_id=candidate_id)
            started = time.perf_counter()
            model = lgb.LGBMClassifier(
                n_estimators=rounds, learning_rate=0.05, num_leaves=63, max_depth=8,
                min_child_samples=40, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True,
            )
            try:
                model.fit(train_features.iloc[indices], train["y"][indices],
                          sample_weight=class_weights(train["y"][indices], mode),
                          eval_set=[(train_features.iloc[selection], train["y"][selection])],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
                metrics = classification_metrics(train["y"][selection],
                                                 model.predict_proba(train_features.iloc[selection]), names)
            except Exception as error:
                record_event(output, "candidate_failed", candidate_id=candidate_id, error_type=type(error).__name__)
                raise
            row = {"candidate_id": candidate_id, "mode": mode, "normal_ratio": ratio, "rows": len(indices),
                   "training_support": dataset_support(train["y"][indices], train["groups"][indices], names),
                   "selection_metrics": metrics, "best_iteration": int(model.best_iteration_),
                   "elapsed_sec": round(time.perf_counter() - started, 4)}
            record_event(output, "candidate_completed", **row)
            experiments.append({**row, "model": model})
    feasible = [row for row in experiments if row["selection_metrics"]["normal_false_alarm_rate"] is not None
                and row["selection_metrics"]["normal_false_alarm_rate"] <= max_false_alarm]
    winner = max(feasible or experiments, key=lambda row: row["selection_metrics"]["macro_f1"])
    model = winner["model"]
    candidate_metrics = classification_metrics(validation["y"], model.predict_proba(validation_features), names)
    champion = load_booster(champion_dir / "robot_pdm_enhanced.txt")
    champion_metrics = classification_metrics(validation["y"], champion.predict(validation["features"]), names)
    promotion = promotion_gate(champion_metrics, candidate_metrics, max_false_alarm)
    if not feasible:
        promotion["eligible"] = False
        promotion["reasons"].append("selection_false_alarm_limit")
    if data_kind != "aihub_recorded":
        promotion["eligible"] = False
        promotion["reasons"].append("data_provenance_not_recorded_aihub")
    report = {
        "schema": "servicerobot.diagnosis.experiment.v2", "seed": seed,
        "lineage": lineage, "environment": environment,
        "search_space": {"normal_ratios": list(normal_ratios), "weight_modes": list(weight_modes),
                         "candidate_count": len(experiments), "rounds": rounds},
        "selection_protocol": "robot_group_holdout", "official_validation_used_for_selection": False,
        "split_audit": {"fit_assets": sorted(set(train["groups"][fit])),
                        "selection_assets": sorted(set(train["groups"][selection])),
                        "official_validation_assets": sorted(set(validation["groups"])), "asset_overlap": 0},
        "dataset_support": {"official_train": dataset_support(train["y"], train["groups"], names),
                            "fit": dataset_support(train["y"][fit], train["groups"][fit], names),
                            "selection": dataset_support(train["y"][selection], train["groups"][selection], names),
                            "official_validation": dataset_support(validation["y"], validation["groups"], names)},
        "experiments": [{key: value for key, value in row.items() if key != "model"} for row in experiments],
        "selected_mode": winner["mode"], "selected_candidate": winner["candidate_id"],
        "selected_normal_ratio": winner["normal_ratio"], "selection_constraint_met": bool(feasible),
        "champion": champion_metrics, "candidate": candidate_metrics,
        "promotion": promotion,
        "probability_calibration": "not_fitted; weighted candidates require calibration before deployment",
    }
    model.booster_.save_model(str(output / "robot_pdm_enhanced.txt"))
    candidate_meta = {"classes": list(range(len(names))), "class_names": names,
                      "n_features": 249, "val_acc": candidate_metrics["accuracy"],
                      "val_macro_f1": candidate_metrics["macro_f1"], "best_iteration": winner["best_iteration"],
                      "dyn": metadata["dyn"], "model_dyn_idx": MODEL_DYN_IDX,
                      "stat": metadata["stat"], "err_map": metadata["err_map"],
                      "selection_protocol": "robot_group_holdout", "automatic_promotion": False,
                      "data_kind": data_kind, "candidate_id": winner["candidate_id"]}
    for name, value in (("robot_pdm_enhanced_meta.json", candidate_meta), ("enhanced_meta.json", metadata),
                        ("experiment_report.json", report)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    comparison = [{"candidate_id": row["candidate_id"],
                   "normal_ratio": "all" if row["normal_ratio"] is None else row["normal_ratio"],
                   "weight_mode": row["mode"], "training_rows": row["rows"], "best_iteration": row["best_iteration"],
                   **{key: row["selection_metrics"][key] for key in
                      ("accuracy", "macro_f1", "normal_false_alarm_rate", "fault_miss_rate")}}
                  for row in experiments]
    pd.DataFrame(comparison).to_csv(output / "selection_results.csv", index=False)
    record_event(output, "run_completed", selected_candidate=winner["candidate_id"], promotion=promotion)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data/experiments/diagnosis-v2")
    parser.add_argument("--champion-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-false-alarm", type=float, default=0.05)
    parser.add_argument("--normal-ratios", nargs="+", type=int, default=[0, 2, 3, 5],
                        help="normal-to-fault caps; 0 keeps all training rows")
    parser.add_argument("--weight-modes", nargs="+", choices=WEIGHT_MODES, default=list(WEIGHT_MODES))
    parser.add_argument("--data-kind", choices=DATA_KINDS, default="unspecified",
                        help="declared data source, not an independently verified provenance claim")
    args = parser.parse_args(argv)
    if args.rounds < 1 or not 0 <= args.max_false_alarm <= 1:
        parser.error("rounds must be positive and false-alarm limit must be between 0 and 1")
    started = time.perf_counter()
    try:
        report = train_candidates(args.data_dir, args.output, args.champion_dir,
                                  args.seed, args.rounds, args.max_false_alarm,
                                  tuple(None if r == 0 else r for r in args.normal_ratios),
                                  tuple(args.weight_modes), args.data_kind)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Diagnosis experiment: {error}\n")
    print(json.dumps({"selected_candidate": report["selected_candidate"], "data_kind": args.data_kind,
                      "candidate": report["candidate"],
                      "promotion": report["promotion"], "elapsed_sec": round(time.perf_counter() - started, 2)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
