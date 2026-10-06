"""Evaluate a saved diagnosis model with per-class and confidence metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnosis_evaluation import classification_metrics
from pdm_runtime import load_booster, make_feature_matrix as feat
from train_diagnosis import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.data_dir / "enhanced_val.npz")
        meta = json.loads((args.model_dir / "robot_pdm_enhanced_meta.json").read_text(encoding="utf-8"))
        dataset_meta = json.loads((args.data_dir / "enhanced_meta.json").read_text(encoding="utf-8"))
        model_dataset_meta = json.loads((args.model_dir / "enhanced_meta.json").read_text(encoding="utf-8"))
        if dataset_meta["err_map"] != dict(zip(meta["class_names"], range(len(meta["class_names"])))):
            raise ValueError("dataset and model label mappings differ")
        for key in ("devtype_map", "mainstate_map", "crowd_map", "dyn", "stat"):
            if dataset_meta[key] != model_dataset_meta[key]:
                raise ValueError(f"dataset and model encodings differ: {key}")
        booster = load_booster(args.model_dir / "robot_pdm_enhanced.txt")
        report = classification_metrics(dataset["y"], booster.predict(dataset["features"]), meta["class_names"])
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Diagnosis evaluation: {error}\n")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
