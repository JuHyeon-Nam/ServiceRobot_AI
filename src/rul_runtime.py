"""
rul_runtime.py — RUL calibration contract and readiness checks.

The current demo does not contain real failure-time labels, so PHM still runs in
heuristic fallback mode. This module makes that boundary explicit and reserves a
stable contract for replacing the heuristic with a supervised RUL/survival model
when asset event/failure labels become available.
"""

from __future__ import annotations

from collections.abc import Iterable


RUL_SCHEMA = "fab.rul.calibration.v1"
RUL_FEATURES = (
    "health",
    "risk_score",
    "trend_slope",
    "vib",
    "batt",
    "temp",
    "warn",
    "severity_code",
)
RUL_LABEL_FIELDS = (
    "asset_id",
    "event_ts",
    "failure_ts",
    "failure_code",
    "censoring",
)
RUL_TARGET = "minutes_to_failure"
MIN_FAILURE_EVENTS = 30
MIN_ASSETS = 3
MIN_CENSORED_WINDOWS = 10

_SEVERITY_CODE = {"정상": 0, "주의": 1, "경고": 2, "위험": 3}
_CENSORED_VALUES = {True, 1, "1", "true", "True", "censored", "CENSORED"}


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rul_feature_vector(health: int, phm: dict, sensors: dict, warn: bool) -> dict:
    """Build the canonical feature vector used by the future RUL model slot."""
    return {
        "health": round(_num(health), 3),
        "risk_score": round(_num(phm.get("risk_score")), 3),
        "trend_slope": round(_num(phm.get("trend_slope")), 3),
        "vib": round(_num(sensors.get("vib")), 3),
        "batt": round(_num(sensors.get("batt")), 3),
        "temp": round(_num(sensors.get("temp")), 3),
        "warn": int(bool(warn)),
        "severity_code": _SEVERITY_CODE.get(phm.get("severity"), 0),
    }


def attach_rul_model_slot(phm: dict, sensors: dict, health: int, warn: bool) -> dict:
    """Attach transparent RUL model metadata to a PHM forecast payload."""
    out = dict(phm)
    out["rul_model"] = {
        "schema": RUL_SCHEMA,
        "mode": "heuristic_fallback",
        "ready_for_supervised": False,
        "feature_names": list(RUL_FEATURES),
        "feature_vector": rul_feature_vector(health, phm, sensors, warn),
        "target": RUL_TARGET,
        "label_join_key": ["asset_id", "event_ts", "failure_ts", "failure_code"],
        "calibration_required": True,
        "reason": "failure-time labels are not included in the replay artifact",
    }
    return out


def rul_calibration_contract() -> dict:
    """Return the data contract needed to replace heuristic RUL with a model."""
    return {
        "schema": RUL_SCHEMA,
        "runtime_mode": "heuristic_fallback",
        "target": RUL_TARGET,
        "feature_fields": list(RUL_FEATURES),
        "label_fields": list(RUL_LABEL_FIELDS),
        "minimum_dataset": {
            "failure_events": MIN_FAILURE_EVENTS,
            "assets": MIN_ASSETS,
            "censored_windows": MIN_CENSORED_WINDOWS,
        },
        "accepted_models": [
            "gradient_boosted_rul_regressor",
            "cox_survival_model",
            "random_survival_forest",
        ],
        "replacement_path": [
            "join telemetry windows to asset failure events",
            "derive minutes_to_failure target or right-censoring flag",
            "train calibrated RUL/survival model",
            "swap heuristic fallback behind the same PHM API contract",
        ],
        "limitation": "Current replay lacks observed failure_ts labels, so RUL estimates remain heuristic.",
    }


def rul_readiness_report(rows: Iterable[dict]) -> dict:
    """Check whether labeled event rows are sufficient for supervised RUL work."""
    records = list(rows)
    missing = {
        field
        for row in records
        for field in RUL_LABEL_FIELDS
        if field not in row or row.get(field) in (None, "")
    }
    assets = {row.get("asset_id") for row in records if row.get("asset_id")}
    censored = sum(row.get("censoring") in _CENSORED_VALUES for row in records)
    failures = sum(
        bool(row.get("failure_ts")) and row.get("censoring") not in _CENSORED_VALUES
        for row in records
    )
    checks = {
        "has_required_fields": not missing,
        "enough_failure_events": failures >= MIN_FAILURE_EVENTS,
        "enough_assets": len(assets) >= MIN_ASSETS,
        "enough_censored_windows": censored >= MIN_CENSORED_WINDOWS,
    }
    return {
        "schema": RUL_SCHEMA,
        "ready_for_supervised": all(checks.values()),
        "rows": len(records),
        "assets": len(assets),
        "failure_events": failures,
        "censored_windows": censored,
        "missing_required_fields": sorted(missing),
        "checks": checks,
        "minimum_dataset": {
            "failure_events": MIN_FAILURE_EVENTS,
            "assets": MIN_ASSETS,
            "censored_windows": MIN_CENSORED_WINDOWS,
        },
    }
