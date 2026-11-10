"""Evaluate a saved diagnosis model with per-class and confidence metrics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from lightgbm.basic import LightGBMError

from diagnosis_evaluation import classification_metrics
from pdm_runtime import FEATURE_BATCH_ROWS, load_booster, make_feature_matrix_batched as feat
from train_diagnosis import (DATA_KINDS, dataset_support, file_sha256, load_dataset,
                              validate_label_coverage, validate_metadata_contract)

ROOT = Path(__file__).resolve().parents[1]


def evaluate_saved_model(data_dir: Path, model_dir: Path, data_kind: str = "unspecified") -> dict:
    """Evaluate without training or modifying model artifacts; retain input evidence."""
    if data_kind not in DATA_KINDS:
        raise ValueError("unknown evaluation data kind")
    files = {"validation_windows": data_dir / "enhanced_val.npz",
             "dataset_metadata": data_dir / "enhanced_meta.json",
             "model": model_dir / "robot_pdm_enhanced.txt",
             "model_metadata": model_dir / "robot_pdm_enhanced_meta.json",
             "model_encoding_metadata": model_dir / "enhanced_meta.json"}
    dataset_meta = json.loads(files["dataset_metadata"].read_text(encoding="utf-8"))
    model_dataset_meta = json.loads(files["model_encoding_metadata"].read_text(encoding="utf-8"))
    meta = json.loads(files["model_metadata"].read_text(encoding="utf-8"))
    names = validate_metadata_contract(dataset_meta, model_dataset_meta, meta)
    dataset = load_dataset(files["validation_windows"], with_features=False)
    validate_label_coverage(dataset["y"], names, "validation")
    features = feat(dataset["X"], dataset["S"])
    del dataset["X"], dataset["S"]
    booster = load_booster(files["model"])
    if booster.num_feature() != 249 or booster.num_model_per_iteration() != len(names):
        raise ValueError("saved booster and class/feature metadata differ")
    report = classification_metrics(dataset["y"], booster.predict(features), names)
    report["evaluation"] = {
        "schema": "servicerobot.diagnosis.evaluation.v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": data_kind, "data_kind_source": "caller_declared_not_independently_verified",
        "dataset_file": "enhanced_val.npz", "assets": len(set(dataset["groups"])),
        "feature_construction": {"mode": "batched", "max_batch_rows": FEATURE_BATCH_ROWS,
                                 "dtype": "float32", "n_features": 249, "raw_sensor_arrays_retained": False},
        "per_class_support": dataset_support(dataset["y"], dataset["groups"], names),
        "input_sha256": {key: file_sha256(path) for key, path in files.items()},
        "code_sha256": {name: file_sha256(ROOT / "src" / name) for name in
                        ("evaluate_enhanced.py", "diagnosis_evaluation.py", "train_diagnosis.py", "pdm_runtime.py")},
        "automatic_promotion": False,
        "limitations": ["source_authenticity_not_verified", "training_validation_asset_overlap_not_checked",
                        "field_performance_not_evaluated"],
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report", type=Path, help="save to a new JSON file; never overwrite")
    parser.add_argument("--data-kind", choices=DATA_KINDS, default="unspecified",
                        help="declared source, not an independently verified provenance claim")
    args = parser.parse_args(argv)
    if args.report is not None and (args.report.exists() or args.report.is_symlink()):
        parser.exit(2, "Diagnosis evaluation: report must be a new file; refusing to overwrite\n")
    try:
        report = evaluate_saved_model(args.data_dir, args.model_dir, args.data_kind)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile, LightGBMError) as error:
        parser.exit(2, f"Diagnosis evaluation: {error}\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
