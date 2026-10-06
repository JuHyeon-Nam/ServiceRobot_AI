"""Select diagnosis candidates on robot holdout; evaluate once on official validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from diagnosis_evaluation import asset_holdout, class_weights, classification_metrics, promotion_gate
from pdm_runtime import MODEL_DYN_IDX, load_booster, make_feature_matrix

ROOT = Path(__file__).resolve().parents[1]


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


def training_rows(labels: np.ndarray, indices: np.ndarray, normal: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    faults = indices[labels[indices] != normal]
    normals = indices[labels[indices] == normal]
    keep = rng.choice(normals, min(len(normals), max(1, len(faults) * 3)), replace=False)
    return np.sort(np.concatenate([faults, keep]))


def train_candidates(data_dir: Path, output: Path, champion_dir: Path,
                     seed: int = 42, rounds: int = 800, max_false_alarm: float = 0.05) -> dict:
    if output.resolve() == champion_dir.resolve():
        raise ValueError("candidate output must not overwrite the serving model directory")
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
    normal = names.index("정상")
    fit, selection = asset_holdout(train["y"], train["groups"], seed)
    train_features = pd.DataFrame(train["features"])
    validation_features = pd.DataFrame(validation["features"])
    experiments = []
    for mode in ("unweighted", "sqrt_balanced", "balanced"):
        indices = training_rows(train["y"], fit, normal, seed) if mode == "unweighted" else fit
        model = lgb.LGBMClassifier(
            n_estimators=rounds, learning_rate=0.05, num_leaves=63, max_depth=8,
            min_child_samples=40, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True,
        )
        model.fit(train_features.iloc[indices], train["y"][indices],
                  sample_weight=class_weights(train["y"][indices], mode),
                  eval_set=[(train_features.iloc[selection], train["y"][selection])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        metrics = classification_metrics(train["y"][selection], model.predict_proba(train_features.iloc[selection]), names)
        experiments.append({"mode": mode, "rows": len(indices), "selection_metrics": metrics,
                            "best_iteration": int(model.best_iteration_), "model": model})
    feasible = [row for row in experiments if row["selection_metrics"]["normal_false_alarm_rate"] is not None
                and row["selection_metrics"]["normal_false_alarm_rate"] <= max_false_alarm]
    winner = max(feasible or experiments, key=lambda row: row["selection_metrics"]["macro_f1"])
    model = winner["model"]
    candidate_metrics = classification_metrics(validation["y"], model.predict_proba(validation_features), names)
    champion_meta = json.loads((champion_dir / "robot_pdm_enhanced_meta.json").read_text(encoding="utf-8"))
    if champion_meta["class_names"] != names or champion_meta["n_features"] != 249:
        raise ValueError("champion and candidate class/feature contracts differ")
    champion = load_booster(champion_dir / "robot_pdm_enhanced.txt")
    champion_metrics = classification_metrics(validation["y"], champion.predict(validation["features"]), names)
    report = {
        "schema": "servicerobot.diagnosis.experiment.v1", "seed": seed,
        "selection_protocol": "robot_group_holdout", "official_validation_used_for_selection": False,
        "split_audit": {"fit_assets": sorted(set(train["groups"][fit])),
                        "selection_assets": sorted(set(train["groups"][selection])),
                        "official_validation_assets": sorted(set(validation["groups"])), "asset_overlap": 0},
        "experiments": [{key: value for key, value in row.items() if key != "model"} for row in experiments],
        "selected_mode": winner["mode"], "selection_constraint_met": bool(feasible),
        "champion": champion_metrics, "candidate": candidate_metrics,
        "promotion": promotion_gate(champion_metrics, candidate_metrics, max_false_alarm),
        "probability_calibration": "not_fitted; weighted candidates require calibration before deployment",
    }
    output.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output / "robot_pdm_enhanced.txt"))
    candidate_meta = {"classes": list(range(len(names))), "class_names": names,
                      "n_features": 249, "val_acc": candidate_metrics["accuracy"],
                      "val_macro_f1": candidate_metrics["macro_f1"], "best_iteration": winner["best_iteration"],
                      "dyn": metadata["dyn"], "model_dyn_idx": MODEL_DYN_IDX,
                      "stat": metadata["stat"], "err_map": metadata["err_map"],
                      "selection_protocol": "robot_group_holdout", "automatic_promotion": False}
    for name, value in (("robot_pdm_enhanced_meta.json", candidate_meta), ("enhanced_meta.json", metadata),
                        ("experiment_report.json", report)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data/experiments/diagnosis-v2")
    parser.add_argument("--champion-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-false-alarm", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.rounds < 1 or not 0 <= args.max_false_alarm <= 1:
        parser.error("rounds must be positive and false-alarm limit must be between 0 and 1")
    started = time.perf_counter()
    try:
        report = train_candidates(args.data_dir, args.output, args.champion_dir,
                                  args.seed, args.rounds, args.max_false_alarm)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Diagnosis experiment: {error}\n")
    print(json.dumps({"selected_mode": report["selected_mode"], "candidate": report["candidate"],
                      "promotion": report["promotion"], "elapsed_sec": round(time.perf_counter() - started, 2)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
