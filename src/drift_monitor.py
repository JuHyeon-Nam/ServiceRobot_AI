"""
drift_monitor.py — live telemetry data-drift monitoring.

The PdM model is only trustworthy while the live robot stream roughly matches
the operating envelope it was built for. This module compares the current AGV
snapshot against a compact reference profile and reports feature-level drift,
fault-rate shift, and an operator-facing recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterable


DEFAULT_REFERENCE_PROFILE = {
    # Healthy service-robot operating envelope used by the simulator/demo layer.
    # Values are intentionally domain-readable so they can be discussed in a
    # model card and later replaced by train/validation distribution statistics.
    "features": {
        "vib": {"mean": 2.2, "std": 1.1, "unit": "mm/s", "label": "vibration"},
        "batt": {"mean": 66.0, "std": 19.0, "unit": "%", "label": "battery"},
        "temp": {"mean": 42.0, "std": 8.0, "unit": "C", "label": "temperature"},
        "health": {"mean": 88.0, "std": 14.0, "unit": "score", "label": "health_index"},
        "conf": {"mean": 0.86, "std": 0.14, "unit": "prob", "label": "confidence"},
    },
    "fault_rate": {"mean": 0.17, "std": 0.08},
}


@dataclass(frozen=True)
class DriftThresholds:
    watch_z: float = 2.0
    drift_z: float = 3.0


class DataDriftMonitor:
    """Compute a small, dependency-free drift report for live AGV telemetry."""

    def __init__(
        self,
        reference_profile: dict[str, Any] | None = None,
        thresholds: DriftThresholds | None = None,
    ):
        self.reference = reference_profile or DEFAULT_REFERENCE_PROFILE
        self.thresholds = thresholds or DriftThresholds()

    def evaluate(self, agvs: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = list(agvs)
        features = self._feature_report(rows)
        fault_rate = self._fault_rate_report(rows)

        max_z = max([0.0, fault_rate["z_abs"], *[f["z_abs"] for f in features.values()]])
        drifted = [name for name, item in features.items() if item["status"] == "drift"]
        watching = [name for name, item in features.items() if item["status"] == "watch"]
        if fault_rate["status"] == "drift":
            drifted.append("fault_rate")
        elif fault_rate["status"] == "watch":
            watching.append("fault_rate")

        status = "ok"
        if drifted:
            status = "drift"
        elif watching:
            status = "watch"

        return {
            "status": status,
            "score": round(max_z, 3),
            "window_size": len(rows),
            "features": features,
            "fault_rate": fault_rate,
            "drifted_features": drifted,
            "watch_features": watching,
            "recommendation": self._recommendation(status, drifted, watching),
        }

    def _feature_report(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for name, ref in self.reference["features"].items():
            values = self._values(rows, name)
            cur_mean = mean(values) if values else 0.0
            cur_std = pstdev(values) if len(values) > 1 else 0.0
            z = (cur_mean - ref["mean"]) / max(ref["std"], 1e-9)
            out[name] = {
                "current_mean": round(cur_mean, 4),
                "current_std": round(cur_std, 4),
                "reference_mean": ref["mean"],
                "reference_std": ref["std"],
                "unit": ref.get("unit", ""),
                "z": round(z, 4),
                "z_abs": round(abs(z), 4),
                "direction": "high" if z > 0 else "low" if z < 0 else "flat",
                "status": self._status(abs(z)),
            }
        return out

    def _fault_rate_report(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        rate = sum(1 for a in rows if a.get("status") == "warn") / max(n, 1)
        ref = self.reference["fault_rate"]
        z = (rate - ref["mean"]) / max(ref["std"], 1e-9)
        return {
            "current": round(rate, 4),
            "reference_mean": ref["mean"],
            "reference_std": ref["std"],
            "z": round(z, 4),
            "z_abs": round(abs(z), 4),
            "status": self._status(abs(z)),
        }

    def _values(self, rows: list[dict[str, Any]], name: str) -> list[float]:
        vals = []
        for row in rows:
            if name in ("vib", "batt", "temp"):
                value = (row.get("sensors") or {}).get(name)
            else:
                value = row.get(name)
            if isinstance(value, (int, float)):
                vals.append(float(value))
        return vals

    def _status(self, z_abs: float) -> str:
        if z_abs >= self.thresholds.drift_z:
            return "drift"
        if z_abs >= self.thresholds.watch_z:
            return "watch"
        return "ok"

    @staticmethod
    def _recommendation(status: str, drifted: list[str], watching: list[str]) -> str:
        if status == "drift":
            keys = ", ".join(drifted[:3])
            return f"live distribution drift detected: review recent telemetry and consider recalibration ({keys})"
        if status == "watch":
            keys = ", ".join(watching[:3])
            return f"distribution shift is building: keep monitoring and sample-label this window ({keys})"
        return "live telemetry is within the reference operating envelope"
