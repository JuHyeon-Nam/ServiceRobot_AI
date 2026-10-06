"""Fetch a pinned NASA archive and extract only the FD001 benchmark files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import requests

SOURCE_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip?download=1"
DATASET_URL = "https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data"
ARCHIVE_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt")
ROOT = Path(__file__).resolve().parents[1]


def prepare_archive(archive: Path, output: Path) -> dict:
    content = archive.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("NASA archive checksum mismatch; data was not extracted")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(archive) as source:
        for name in FILES:
            info = source.getinfo(name)
            if info.file_size > 20_000_000:
                raise ValueError("unexpected dataset file size")
            data = source.read(name)
            target = output / name
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"existing dataset differs: {target}")
            target.write_bytes(data)
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {
        "dataset": "NASA C-MAPSS FD001", "kind": "engine_degradation_simulation",
        "source_url": SOURCE_URL, "dataset_url": DATASET_URL,
        "archive_sha256": digest, "files": files,
        "citation": "Saxena, Goebel, Simon, Eklund (2008), Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/cmapss")
    args = parser.parse_args()
    archive = args.archive or args.output / "CMAPSSData.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(SOURCE_URL, timeout=(10, 60))
        response.raise_for_status()
        if len(response.content) > 50_000_000:
            raise ValueError("unexpected archive size")
        if hashlib.sha256(response.content).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("NASA archive checksum mismatch")
        archive.write_bytes(response.content)
    print(json.dumps(prepare_archive(archive, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
