"""Header/byte inspection on synthetic NPZs, never real model-performance evidence."""
import io
import json
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import inspect_diagnosis_dataset as inspector
from pdm_runtime import make_feature_matrix


def arrays(rows=3):
    return {"X": np.zeros((rows, 30, 7), dtype=np.float32),
            "S": np.zeros((rows, 9), dtype=np.float64), "y": np.zeros(rows, dtype=np.int64),
            "groups": np.full(rows, "robot-a")}


def npy_bytes(array, version=(1, 0)):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=version, allow_pickle=True)
    return stream.getvalue()


def write_npz(path, rows, version=(1, 0), compression=zipfile.ZIP_DEFLATED, extras=()):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for key, array in rows.items():
            archive.writestr(f"{key}.npy", npy_bytes(array, version))
        for name, content in extras:
            archive.writestr(name, content)


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
@pytest.mark.parametrize("fortran", [False, True])
def test_byte_counts_match_real_arrays_without_loading_payloads(tmp_path, monkeypatch, version, compression, fortran):
    rows = arrays(5000)
    rows["X"] = np.asfortranarray(rows["X"]) if fortran else rows["X"]
    path = tmp_path / "windows.npz"
    write_npz(path, rows, version, compression, extras=[("notes.txt", b"not a required array")])
    before = path.read_bytes()
    reads = []
    original_read = inspector._HeaderReader.read

    def forbidden(*args, **kwargs):
        pytest.fail("resource inspection must not load or allocate an array payload")

    def tracked_read(reader, size):
        reads.append(size)
        return original_read(reader, size)

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    monkeypatch.setattr(np.lib.format, "read_array", forbidden)
    monkeypatch.setattr(inspector._HeaderReader, "read", tracked_read)
    report = inspector.inspect_dataset(path)
    assert report["schema"] == "servicerobot.diagnosis.resources.v1"
    assert report["rows"] == 5000 and report["header_contract_compatible"]
    assert report["training_readiness"] == "not_evaluated"
    assert not report["model_performance_evaluated"] and not report["array_payloads_loaded"]
    assert not report["full_zip_crc_checked"]
    assert report["npz_file_bytes"] == len(before)
    assert report["zip"]["member_count"] == 5 and report["zip"]["other_member_count"] == 1
    with zipfile.ZipFile(path) as archive:
        assert report["zip"]["member_compressed_bytes"] == sum(info.compress_size for info in archive.infolist())
        assert report["zip"]["member_uncompressed_bytes"] == sum(info.file_size for info in archive.infolist())
    for key, raw in rows.items():
        entry = report["arrays"][key]
        assert entry["shape"] == list(raw.shape) and entry["dtype"] == raw.dtype.str
        assert entry["npy_version"] == list(version)
        assert entry["array_payload_bytes"] == raw.nbytes
        assert entry["zip_member_uncompressed_bytes"] == raw.nbytes + entry["npy_header_bytes"]
        assert entry["npy_header_bytes"] <= inspector.MAX_HEADER_BYTES + 12
    assert report["arrays"]["X"]["fortran_order"] == fortran
    assert sum(reads) == sum(entry["npy_header_bytes"] for entry in report["arrays"].values())
    assert report["estimates"]["required_array_payload_bytes"] == sum(raw.nbytes for raw in rows.values())
    assert report["estimates"]["raw_sensor_payload_bytes"] == rows["X"].nbytes + rows["S"].nbytes
    assert report["estimates"]["feature_matrix"]["bytes"] == 5000 * 249 * 4
    assert not report["estimates"]["is_peak_ram_estimate"]
    assert path.read_bytes() == before
    json.dumps(report, allow_nan=False)


def test_feature_estimate_matches_the_current_runtime_contract(tmp_path):
    rows = arrays()
    path = tmp_path / "windows.npz"
    write_npz(path, rows)
    report = inspector.inspect_dataset(path)
    actual = make_feature_matrix(rows["X"], rows["S"])
    assert report["estimates"]["feature_matrix"] == {
        "rows": 3, "columns": actual.shape[1], "dtype": str(actual.dtype), "bytes": actual.nbytes}


@pytest.mark.parametrize("changes", [
    {"X": np.zeros((3, 29, 7))}, {"S": np.zeros((3, 8))}, {"y": np.zeros((3, 1), dtype=int)},
    {"groups": np.array(["robot"])}, {"X": np.zeros((3, 30, 7), dtype=bool)},
    {"S": np.zeros((3, 9), dtype=complex)}, {"y": np.zeros(3, dtype=float)},
    {"groups": np.arange(3)}, {"groups": np.array([None] * 3)},
])
def test_incompatible_shapes_and_dtypes_are_rejected(tmp_path, changes):
    rows = arrays()
    rows.update(changes)
    path = tmp_path / "invalid.npz"
    write_npz(path, rows)
    with pytest.raises(ValueError):
        inspector.inspect_dataset(path)


def test_empty_windows_are_not_reported_as_compatible(tmp_path):
    path = tmp_path / "empty.npz"
    write_npz(path, arrays(0))
    with pytest.raises(ValueError, match="nonempty"):
        inspector.inspect_dataset(path)


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_required_member_names_must_be_unambiguous(tmp_path, kind):
    path = tmp_path / "ambiguous.npz"
    rows = arrays()
    if kind == "missing":
        rows.pop("groups")
        write_npz(path, rows)
    else:
        with pytest.warns(UserWarning, match="Duplicate name"):
            write_npz(path, rows, extras=[("groups.npy", npy_bytes(rows["groups"]))])
    with pytest.raises(ValueError, match="exactly one groups.npy"):
        inspector.inspect_dataset(path)


@pytest.mark.parametrize("payload,message", [
    (b"not-npy", "magic"), (np.lib.format.magic(3, 0), "unsupported"),
    (np.lib.format.magic(2, 0) + struct.pack("<I", 2**32 - 1), "read budget"),
])
def test_bad_or_oversized_headers_fail_without_reading_an_array(tmp_path, payload, message):
    path = tmp_path / "bad-header.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("X.npy", payload)
    with pytest.raises(ValueError, match=message):
        inspector.inspect_dataset(path)


def test_huge_declared_shape_does_not_trigger_a_huge_allocation(tmp_path, monkeypatch):
    stream = io.BytesIO()
    np.lib.format.write_array_header_2_0(stream, {"shape": (10**15, 30, 7),
                                               "fortran_order": False, "descr": "<f4"})
    path = tmp_path / "header-only.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("X.npy", stream.getvalue())

    def forbidden(*args, **kwargs):
        pytest.fail("a declared shape must not cause payload allocation")

    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="declared array size differs"):
        inspector.inspect_dataset(path)


def test_trailing_bytes_are_not_accepted_as_an_exact_array_payload(tmp_path):
    path = tmp_path / "trailing.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("X.npy", npy_bytes(arrays()["X"]) + b"extra")
    with pytest.raises(ValueError, match="declared array size differs"):
        inspector.inspect_dataset(path)


def test_compatible_headers_do_not_assert_sensor_or_robot_value_validity(tmp_path):
    rows = arrays()
    rows["X"][:] = np.nan
    rows["y"][:] = -1
    rows["groups"][:] = ""
    path = tmp_path / "bad-values.npz"
    write_npz(path, rows)
    report = inspector.inspect_dataset(path)
    assert report["header_contract_compatible"]
    assert report["training_readiness"] == "not_evaluated"
    assert "sensor_values_labels_and_robot_ids_not_read_or_validated" in report["limitations"]
    assert "training_validation_robot_overlap_not_checked" in report["limitations"]


def test_header_compatibility_does_not_assert_full_payload_crc_integrity(tmp_path):
    path = tmp_path / "corrupt-payload.npz"
    rows = arrays(5000)
    write_npz(path, rows, compression=zipfile.ZIP_STORED)
    payload = npy_bytes(rows["X"])
    original = path.read_bytes()
    assert original.count(payload) == 1
    corrupted = bytearray(original)
    corrupted[original.index(payload) + len(payload) - 1] ^= 1
    path.write_bytes(corrupted)
    report = inspector.inspect_dataset(path)
    assert report["header_contract_compatible"] and not report["full_zip_crc_checked"]
    assert report["training_readiness"] == "not_evaluated"
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(zipfile.BadZipFile, match="CRC"):
            archive.read("X.npy")


def test_cli_json_matches_stdout_and_preserves_input(tmp_path, capsys):
    path, report_path = tmp_path / "data.npz", tmp_path / "reports/resources.json"
    write_npz(path, arrays())
    before = path.read_bytes()
    assert inspector.main([str(path), "--report", str(report_path)]) == 0
    assert json.loads(report_path.read_text()) == json.loads(capsys.readouterr().out)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["file", "input", "directory", "symlink", "dangling_symlink"])
def test_cli_never_overwrites_an_existing_report_or_input(tmp_path, monkeypatch, kind):
    path, target = tmp_path / "data.npz", tmp_path / "report.json"
    write_npz(path, arrays())
    if kind == "input":
        target = path
    elif kind == "directory":
        target.mkdir()
    elif kind in {"symlink", "dangling_symlink"}:
        target.symlink_to(path if kind == "symlink" else tmp_path / "missing")
    else:
        target.write_bytes(b"existing report")
    before = path.read_bytes()
    target_before = target.read_bytes() if target.is_file() else None

    def forbidden(*args):
        pytest.fail("existing report must be rejected before input inspection")

    monkeypatch.setattr(inspector, "inspect_dataset", forbidden)
    with pytest.raises(SystemExit) as error:
        inspector.main([str(path), "--report", str(target)])
    assert error.value.code == 2
    assert path.read_bytes() == before
    if target_before is not None:
        assert target.read_bytes() == target_before


def test_report_created_during_inspection_is_preserved(tmp_path, monkeypatch):
    path, target = tmp_path / "data.npz", tmp_path / "report.json"
    write_npz(path, arrays())
    original = inspector.inspect_dataset

    def concurrent_writer(path):
        report = original(path)
        target.write_bytes(b"other result")
        return report

    monkeypatch.setattr(inspector, "inspect_dataset", concurrent_writer)
    with pytest.raises(SystemExit) as error:
        inspector.main([str(path), "--report", str(target)])
    assert error.value.code == 2
    assert target.read_bytes() == b"other result"


def test_missing_input_fails_without_creating_a_report(tmp_path):
    target = tmp_path / "resources.json"
    with pytest.raises(SystemExit) as error:
        inspector.main([str(tmp_path / "missing.npz"), "--report", str(target)])
    assert error.value.code == 2
    assert not target.exists()
