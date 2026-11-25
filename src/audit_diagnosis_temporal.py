"""Pre-execution temporal proposal audit: descriptors only, no training or model evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from math import prod
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

from diagnosis_evaluation import class_support_audit, validate_support_minima
from diagnosis_temporal import _utc_ns, history_span_bound, purged_temporal_holdout
from inspect_diagnosis_dataset import _HeaderReader, inspect_dataset
from train_diagnosis import DATA_KINDS, ROOT, file_sha256, validate_metadata_contract

MAX_DESCRIPTOR_BYTES = 128 * 1024 * 1024


def load_descriptors(path):
    layout = inspect_dataset(path)
    readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    total = 0
    with zipfile.ZipFile(path) as archive:
        for name in ("y", "groups", "target_times"):
            matches = [m for m in archive.infolist() if m.filename == name + ".npy"]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one {name}.npy descriptor")
            member = matches[0]
            with archive.open(member) as stream:
                reader = _HeaderReader(stream)
                version = np.lib.format.read_magic(reader)
                if version not in readers:
                    raise ValueError("unsupported descriptor NPY version")
                shape, _, dtype = readers[version](reader, max_header_size=10000)
            allowed = "iu" if name == "y" else "US"
            if shape != (layout["rows"],) or dtype.hasobject or dtype.kind not in allowed or not dtype.itemsize:
                raise ValueError(f"invalid {name} descriptor shape or dtype")
            payload = prod(shape) * dtype.itemsize
            if reader.position + payload != member.file_size:
                raise ValueError("descriptor size differs from NPY header")
            total += payload
    if total > MAX_DESCRIPTOR_BYTES:
        raise ValueError("descriptor payload exceeds preparation memory budget")
    with np.load(path, allow_pickle=False) as arrays:
        rows = {key: arrays[key] for key in ("y", "groups", "target_times")}
    return rows, layout, total


def audit_temporal(data_dir, champion_dir, *, cutoff, min_selection_rows=20, min_selection_assets=2,
                   data_kind="unspecified"):
    _utc_ns(cutoff)
    policy = validate_support_minima(min_selection_rows, min_selection_assets)
    if data_kind not in DATA_KINDS:
        raise ValueError("unknown data kind")
    start = time.monotonic()
    path, meta_path = data_dir / "enhanced_train.npz", data_dir / "enhanced_meta.json"
    paths = {"enhanced_train.npz": path, "enhanced_meta.json": meta_path,
             "reference_dataset_meta": champion_dir / "enhanced_meta.json",
             "reference_model_meta": champion_dir / "robot_pdm_enhanced_meta.json"}
    hashes = {name: file_sha256(p) for name, p in paths.items()}
    metadata = json.loads(meta_path.read_bytes())
    if not isinstance(metadata, dict):
        raise ValueError("temporal metadata must be an object")
    scope = metadata.get("build_scope", {})
    if (not isinstance(scope, dict) or scope.get("mode") != "training_only"
            or scope.get("official_validation_read") is not False
            or scope.get("official_validation_windows_created") is not False):
        raise ValueError("temporal preparation requires declared Training-only metadata")
    names = validate_metadata_contract(metadata,
        json.loads(paths["reference_dataset_meta"].read_bytes()), json.loads(paths["reference_model_meta"].read_bytes()))
    bound = history_span_bound(metadata)
    rows, layout, descriptor_bytes = load_descriptors(path)
    if (rows["y"] < 0).any() or (rows["y"] >= len(names)).any():
        raise ValueError("unknown training class")
    fit, selection, plan = purged_temporal_holdout(rows["groups"], rows["target_times"],
        cutoff=cutoff, history_span_ns=bound["history_span_upper_bound_ns"])
    fit_support = class_support_audit(rows["y"][fit], rows["groups"][fit], names)
    selection_support = class_support_audit(rows["y"][selection], rows["groups"][selection], names, **policy)
    if hashes != {name: file_sha256(p) for name, p in paths.items()}:
        raise ValueError("input files changed during temporal preparation")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    code_files = ("audit_diagnosis_temporal.py", "diagnosis_temporal.py", "inspect_diagnosis_dataset.py",
                  "diagnosis_evaluation.py", "train_diagnosis.py", "build_enhanced_dataset.py")
    return {"schema": "servicerobot.diagnosis.temporal_preparation.v1",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "training_descriptors_proposal_only_not_model_evaluation",
            "protocol_status": "proposal_not_approved_or_selected", "requires_user_scope_approval": True,
            "authorized_for_training": False, "training_readiness": "not_evaluated",
            "data_kind": data_kind, "data_kind_source": "caller_declared_not_independently_verified",
            "cutoff_source": "single_caller_declared_cutoff_no_label_based_search",
            "history_bound": bound, "plan": plan, "fit_support": fit_support,
            "selection_support": selection_support,
            "preparation_checks_passed": bool(plan["both_sides_nonempty"] and
                fit_support["sufficient_support"] and selection_support["sufficient_support"]),
            "layout": layout, "descriptor_payload_bytes": descriptor_bytes,
            "official_validation_read": False, "sensor_array_payloads_loaded": False,
            "full_npz_crc_checked": False, "feature_matrix_created": False,
            "model_trained": False, "model_performance_evaluated": False, "official_validation_metrics": None,
            "lineage": {"sha256": hashes, "code_sha256": {name: file_sha256(ROOT / "src" / name) for name in code_files}},
            "resources": {"elapsed_seconds": round(time.monotonic() - start, 3),
                          "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024},
            "limitations": ["preparation_pass_is_not_user_consent_training_readiness_or_model_performance",
                            "metadata_history_bound_not_independently_verified_raw_history_intervals",
                            "only_descriptor_arrays_crc_checked; sensor_array_bodies_not_read",
                            "descriptor_budget_is_not_a_peak_ram_bound",
                            "frozen_metadata_may_include_historical_validation_information",
                            "input_hashes_do_not_prove_atomic_snapshot_or_source_authenticity",
                            "not_new_robot_generalization_or_official_validation"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--champion-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--cutoff", required=True, help="one timezone-aware ISO8601 cutoff; no search")
    parser.add_argument("--min-selection-rows", type=int, default=20)
    parser.add_argument("--min-selection-assets", type=int, default=2)
    parser.add_argument("--data-kind", choices=DATA_KINDS, default="unspecified")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        parser.exit(2, "Temporal preparation: report must be a new file\n")
    try:
        report = audit_temporal(args.data_dir, args.champion_dir, cutoff=args.cutoff,
            min_selection_rows=args.min_selection_rows, min_selection_assets=args.min_selection_assets,
            data_kind=args.data_kind)
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Temporal preparation failed: {error}\n")
    print(encoded)
    return 0 if report["preparation_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
