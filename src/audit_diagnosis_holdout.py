"""Audit Training-only selection support without features, model fitting, or official Validation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from numbers import Integral
from pathlib import Path
import zipfile

from diagnosis_evaluation import asset_holdout, class_holdout_bounds, class_support_audit, validate_support_minima
from train_diagnosis import (DATA_KINDS, ROOT, dataset_support, file_sha256, load_dataset,
                             validate_label_coverage, validate_metadata_contract)


def _load_training_support(data_dir, champion_dir, min_selection_rows, min_selection_assets, data_kind):
    policy = validate_support_minima(min_selection_rows, min_selection_assets)
    if data_kind not in DATA_KINDS:
        raise ValueError("unknown data kind")
    meta_path = data_dir / "enhanced_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    names = validate_metadata_contract(metadata,
        json.loads((champion_dir / "enhanced_meta.json").read_text(encoding="utf-8")),
        json.loads((champion_dir / "robot_pdm_enhanced_meta.json").read_text(encoding="utf-8")))
    path = data_dir / "enhanced_train.npz"
    rows = load_dataset(path, with_features=False, keep_raw=False)
    if (rows["y"] < 0).any() or (rows["y"] >= len(names)).any():
        raise ValueError("unknown training class")
    report = {"schema": "servicerobot.diagnosis.holdout_support.v1",
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_scope": "training_robot_holdout_support_only", "selection_support_policy": policy,
              "data_kind": data_kind, "data_kind_source": "caller_declared_not_independently_verified",
              "training_rows": len(rows["y"]),
              "training_support": dataset_support(rows["y"], rows["groups"], names),
              "class_holdout_bounds": class_holdout_bounds(rows["y"], rows["groups"], names,
                  min_selection_rows=policy["min_rows_per_class"],
                  min_selection_assets=policy["min_assets_per_class"]),
              "official_validation_read": False, "feature_matrix_created": False,
              "raw_sensor_arrays_retained": False, "model_trained": False, "model_performance_evaluated": False,
              "training_readiness": "not_evaluated_beyond_selection_support",
              "lineage": {"sha256": {"enhanced_train.npz": file_sha256(path),
                                      "enhanced_meta.json": file_sha256(meta_path)},
                          "reference_metadata_sha256": {name: file_sha256(champion_dir / name) for name in
                              ("enhanced_meta.json", "robot_pdm_enhanced_meta.json")},
                          "code_sha256": {name: file_sha256(ROOT / "src" / name) for name in
                              ("audit_diagnosis_holdout.py", "diagnosis_evaluation.py", "train_diagnosis.py")}},
              "limitations": ["minima_are_engineering_policy_not_statistical_confidence_or_power",
                              "windows_not_independent_failure_episodes",
                              "continuity_policy_and_full_npz_crc_not_audited",
                              "input_hashes_do_not_prove_an_atomic_snapshot",
                              "frozen_metadata_may_include_historical_validation_information",
                              "not_a_model_evaluation_or_promotion_decision"]}
    return rows, names, report


def _audit_split(rows, names, template, seed):
    report = deepcopy(template)
    report.update({"seed": seed, "support_gate_passed": False,
                   "selection_support_audit": None, "holdout": None, "issues": []})
    policy = report["selection_support_policy"]
    try:
        validate_label_coverage(rows["y"], names, "training")
        fit, selection = asset_holdout(rows["y"], rows["groups"], seed, require_all_classes=True)
    except ValueError as error:
        report["issues"].append({"code": "holdout_unavailable", "detail": str(error)})
        return report
    support = class_support_audit(rows["y"][selection], rows["groups"][selection], names, **policy)
    report["selection_support_audit"] = support
    report["support_gate_passed"] = support["sufficient_support"]
    report["holdout"] = {"fit_rows": len(fit), "selection_rows": len(selection),
                         "fit_assets": sorted(set(rows["groups"][fit])),
                         "selection_assets": sorted(set(rows["groups"][selection])),
                         "fit_support": dataset_support(rows["y"][fit], rows["groups"][fit], names),
                         "asset_overlap": len(set(rows["groups"][fit]) & set(rows["groups"][selection]))}
    report["issues"] = list(support["issues"])
    return report


def audit_holdout(data_dir, champion_dir, *, seed=42, min_selection_rows=20,
                  min_selection_assets=2, data_kind="unspecified"):
    rows, names, report = _load_training_support(data_dir, champion_dir, min_selection_rows,
                                                min_selection_assets, data_kind)
    return _audit_split(rows, names, report, seed)


def _validate_starting_seeds(seeds):
    if (not isinstance(seeds, (list, tuple)) or not 1 <= len(seeds) <= 64
            or any(not isinstance(seed, Integral) or isinstance(seed, bool)
                   or not 0 <= seed <= 2**32 - 64 for seed in seeds)):
        raise ValueError("provide 1 to 64 distinct integer starting seeds in [0, 2**32 - 64]")
    values = [int(seed) for seed in seeds]
    if len(set(values)) != len(values):
        raise ValueError("starting seeds must be distinct")
    return values


def audit_repeated_holdouts(data_dir, champion_dir, *, seeds=(42, 43, 44, 45, 46),
                            min_selection_rows=20, min_selection_assets=2, data_kind="unspecified"):
    """Inspect every declared request; never rank seeds or treat repeats as independent samples."""
    seeds = _validate_starting_seeds(seeds)
    rows, names, template = _load_training_support(data_dir, champion_dir, min_selection_rows,
                                                  min_selection_assets, data_kind)
    runs = [_audit_split(rows, names, template, seed) for seed in seeds]
    generated = [run for run in runs if run["holdout"] is not None]
    partitions = {}
    for run in generated:
        key = tuple(run["holdout"]["selection_assets"])
        partitions.setdefault(key, []).append(run["seed"])
    ranges = {}
    for name in names:
        counts = [run["selection_support_audit"]["per_class"][name] for run in generated]
        ranges[name] = {f"{unit}_{bound}": (function(row[unit] for row in counts) if counts else None)
                        for unit in ("rows", "assets") for bound, function in (("min", min), ("max", max))}
        policy = template["selection_support_policy"]
        ranges[name].update({"evaluated_requests": len(generated),
            "requests_passing_minima": sum(row["rows"] >= policy["min_rows_per_class"] and
                                           row["assets"] >= policy["min_assets_per_class"] for row in counts)})
    report = deepcopy(template)
    report.update({"schema": "servicerobot.diagnosis.holdout_support_repeats.v1",
                   "evaluation_scope": "repeated_training_robot_holdout_support_only",
                   "starting_seeds": seeds,
                   "seed_semantics": "starting_seeds_for_existing_up_to_64_attempt_class_coverage_search",
                   "requested_seed_count": len(seeds), "generated_holdout_count": len(generated),
                   "unavailable_holdout_count": len(runs) - len(generated),
                   "passing_request_count": sum(run["support_gate_passed"] for run in runs),
                   "support_gate_passed": all(run["support_gate_passed"] for run in runs),
                   "distinct_selection_asset_sets": len(partitions),
                   "partition_requests": [{"selection_assets": list(assets), "starting_seeds": values}
                                          for assets, values in partitions.items()],
                   "selection_support_ranges": ranges, "runs": runs,
                   "best_seed_selected": False})
    report["limitations"].extend(["support_ranges_are_not_confidence_intervals_or_model_metrics",
                                   "different_starting_seeds_can_repeat_the_same_robot_partition",
                                   "class_coverage_search_is_not_uniform_group_cross_validation"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--champion-dir", type=Path, default=ROOT / "data/processed")
    seed_options = parser.add_mutually_exclusive_group()
    # Python 3.11 can ignore an explicit default value in a mutually exclusive group.
    seed_options.add_argument("--seed", type=int, default=None, help="single starting seed (default: 42)")
    seed_options.add_argument("--seeds", type=int, nargs="+",
                              help="audit all declared starting seeds; no best-seed selection; at most 64")
    parser.add_argument("--min-selection-rows", type=int, default=20)
    parser.add_argument("--min-selection-assets", type=int, default=2)
    parser.add_argument("--data-kind", choices=DATA_KINDS, default="unspecified")
    parser.add_argument("--report", required=True, type=Path, help="new JSON file; never overwrite")
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        parser.exit(2, "Holdout support: report must be a new file\n")
    try:
        options = {"min_selection_rows": args.min_selection_rows,
                   "min_selection_assets": args.min_selection_assets, "data_kind": args.data_kind}
        seed = 42 if args.seed is None else args.seed
        report = (audit_holdout(args.data_dir, args.champion_dir, seed=seed, **options) if args.seeds is None
                  else audit_repeated_holdouts(args.data_dir, args.champion_dir, seeds=args.seeds, **options))
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Holdout support failed: {error}\n")
    print(encoded)
    return 0 if report["support_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
