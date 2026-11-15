"""Inspect diagnosis NPZ layout and byte counts without loading sensor arrays."""
from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path
import zipfile

import numpy as np

REQUIRED_ARRAYS = ("X", "S", "y", "groups")
MAX_HEADER_BYTES = 10_000
FEATURE_COUNT = 249


class _HeaderReader:
    """Bound logical reads before NumPy can request an oversized declared header."""

    def __init__(self, stream):
        self.stream, self.position = stream, 0

    def read(self, size):
        if size < 0 or size > MAX_HEADER_BYTES + 12 - self.position:
            raise ValueError("NPY header exceeds the inspection read budget")
        data = self.stream.read(size)
        self.position += len(data)
        return data


def inspect_dataset(path: Path) -> dict:
    """Header compatibility is not data readiness, source verification, or a RAM benchmark."""
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for name in REQUIRED_ARRAYS:
            matches = [member for member in members if member.filename == f"{name}.npy"]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one {name}.npy member, found {len(matches)}")
            member = matches[0]
            with archive.open(member) as stream:
                reader = _HeaderReader(stream)
                version = np.lib.format.read_magic(reader)
                if version not in readers:
                    raise ValueError(f"NPY version {version} is unsupported by this header inspector")
                shape, fortran_order, dtype = readers[version](reader, max_header_size=MAX_HEADER_BYTES)
            if dtype.hasobject:
                raise ValueError(f"{name} object/pickle dtype is not supported")
            if any(type(dimension) is not int or dimension < 0 for dimension in shape):
                raise ValueError(f"{name} has an invalid declared shape")
            allowed = "iuf" if name in ("X", "S") else "iu" if name == "y" else "US"
            if dtype.kind not in allowed or not dtype.itemsize:
                raise ValueError(f"{name} has an incompatible declared dtype")
            payload_bytes = prod(shape) * dtype.itemsize
            if reader.position + payload_bytes != member.file_size:
                raise ValueError(f"{name} declared array size differs from ZIP member size")
            arrays[name] = {"shape": list(shape), "dtype": dtype.str, "fortran_order": fortran_order,
                            "npy_version": list(version), "npy_header_bytes": reader.position,
                            "array_payload_bytes": payload_bytes,
                            "zip_member_compressed_bytes": member.compress_size,
                            "zip_member_uncompressed_bytes": member.file_size}
        shapes = {name: tuple(info["shape"]) for name, info in arrays.items()}
        if len(shapes["y"]) != 1 or not shapes["y"][0]:
            raise ValueError("y must declare nonempty one-dimensional labels")
        rows = shapes["y"][0]
        if shapes != {"X": (rows, 30, 7), "S": (rows, 9), "y": (rows,), "groups": (rows,)}:
            raise ValueError("declared arrays do not match the diagnosis window/alignment contract")
        zip_bytes = {"member_compressed_bytes": sum(member.compress_size for member in members),
                     "member_uncompressed_bytes": sum(member.file_size for member in members),
                     "member_count": len(members), "other_member_count": len(members) - len(REQUIRED_ARRAYS)}
    return {"schema": "servicerobot.diagnosis.resources.v1", "file": str(path.resolve()),
            "inspection": "npy_headers_and_zip_directory_only", "header_contract_compatible": True,
            "training_readiness": "not_evaluated", "model_performance_evaluated": False,
            "array_payloads_loaded": False, "full_zip_crc_checked": False,
            "npz_file_bytes": path.stat().st_size, "zip": zip_bytes, "rows": rows, "arrays": arrays,
            "estimates": {"required_array_payload_bytes": sum(info["array_payload_bytes"] for info in arrays.values()),
                          "raw_sensor_payload_bytes": arrays["X"]["array_payload_bytes"] + arrays["S"]["array_payload_bytes"],
                          "feature_matrix": {"rows": rows, "columns": FEATURE_COUNT, "dtype": "float32",
                                             "bytes": rows * FEATURE_COUNT * np.dtype("float32").itemsize},
                          "is_peak_ram_estimate": False},
            "limitations": ["sensor_values_labels_and_robot_ids_not_read_or_validated",
                            "training_validation_robot_overlap_not_checked", "metadata_model_contract_not_checked",
                            "full_zip_crc_not_checked", "source_authenticity_and_input_snapshot_not_verified",
                            "not_peak_ram_or_allocated_disk_measurement",
                            "numpy_lightgbm_temporaries_models_and_other_files_excluded"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="enhanced_train.npz or enhanced_val.npz")
    parser.add_argument("--report", type=Path, help="save to a new JSON file; never overwrite")
    args = parser.parse_args(argv)
    if args.report is not None and (args.report.exists() or args.report.is_symlink()):
        parser.exit(2, "Diagnosis inspection: report must be a new file; refusing to overwrite\n")
    try:
        report = inspect_dataset(args.dataset)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, NotImplementedError) as error:
        parser.exit(2, f"Diagnosis inspection: {error}\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
