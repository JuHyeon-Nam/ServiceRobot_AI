"""Read-only benchmark evidence. No engine model is loaded into AGV serving."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "docs/benchmarks/cmapss_fd001.json"


def load_report(path: Path | None = None) -> dict:
    path = path or Path(os.environ.get("RUL_BENCHMARK_REPORT", DEFAULT_REPORT))
    if path.stat().st_size > 2_000_000:
        raise ValueError("benchmark report too large")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report["schema"] != "servicerobot.cmapss.benchmark.v1"
            or report["data_kind"] != "engine_degradation_simulation"
            or report["target_unit"] != "cycles"
            or report["serving_enabled_for_agv"] is not False):
        raise ValueError("incompatible benchmark provenance")
    for metrics in (report["baseline"], report["test_metrics"]):
        for key in ("mae_cycles", "rmse_cycles"):
            if not math.isfinite(metrics[key]) or metrics[key] < 0:
                raise ValueError("invalid benchmark metrics")
    coverage = report["test_metrics"]["interval_coverage"]
    if not math.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("invalid coverage")
    engines = report["engines"]
    ids = [engine["unit"] for engine in engines]
    if not ids or len(set(ids)) != len(ids) or len(ids) != report["protocol"]["test_engines"]:
        raise ValueError("incompatible benchmark engines")
    for engine in engines:
        history = engine["history"]
        if not history or history[-1]["cycle"] != engine["last_cycle"]:
            raise ValueError("invalid benchmark history")
        previous = 0
        for row in history:
            if row["cycle"] <= previous:
                raise ValueError("unordered benchmark history")
            previous = row["cycle"]
            values = [row[key] for key in ("true_rul", "prediction", "lower", "upper")]
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError("invalid benchmark values")
            if not row["lower"] <= row["prediction"] <= row["upper"]:
                raise ValueError("invalid benchmark interval")
    return report


def summary(report: dict) -> dict:
    return {"available": True, **{key: value for key, value in report.items() if key != "engines"},
            "engine_ids": [engine["unit"] for engine in report["engines"]]}
