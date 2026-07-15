"""
dataset_quality.py — Robotics dataset QA and governance metrics.

This module turns raw robotics samples into operational data-quality indicators:
schema completeness, annotation coverage, QA pass rate, ingest success rate, and
rework rate. The goal is to make the PdM demo useful not only as an inference
system, but also as a small robotics data pipeline prototype.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "sample_id",
    "robot_id",
    "timestamp",
    "modality",
    "storage_uri",
    "annotation",
)

VALID_MODALITIES = {"sensor", "image", "video", "egocentric", "telemetry"}
VALID_QA_STATUS = {"pass", "fail", "needs_review"}


@dataclass(frozen=True)
class QualityIssue:
    sample_id: str
    field: str
    reason: str


@dataclass
class QualityReport:
    total: int
    valid: int = 0
    invalid: int = 0
    annotated: int = 0
    qa_passed: int = 0
    needs_review: int = 0
    rework_required: int = 0
    stored: int = 0
    by_modality: dict[str, int] = field(default_factory=dict)
    issues: list[QualityIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        denom = max(self.total, 1)
        annotated_denom = max(self.annotated, 1)
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "annotated": self.annotated,
            "stored": self.stored,
            "qa_passed": self.qa_passed,
            "needs_review": self.needs_review,
            "rework_required": self.rework_required,
            "schema_valid_rate": round(self.valid / denom, 4),
            "annotation_coverage": round(self.annotated / denom, 4),
            "qa_pass_rate": round(self.qa_passed / annotated_denom, 4),
            "ingest_success_rate": round(self.stored / denom, 4),
            "rework_rate": round(self.rework_required / denom, 4),
            "by_modality": dict(sorted(self.by_modality.items())),
            "issues": [issue.__dict__ for issue in self.issues[:50]],
        }


class RoboticsDataQualityMonitor:
    """Validate robotics learning-data records and compute governance metrics."""

    def __init__(
        self,
        required_fields: Iterable[str] = REQUIRED_FIELDS,
        valid_modalities: Iterable[str] = VALID_MODALITIES,
    ):
        self.required_fields = tuple(required_fields)
        self.valid_modalities = set(valid_modalities)

    def validate_record(self, record: dict[str, Any]) -> list[QualityIssue]:
        sample_id = str(record.get("sample_id") or "<missing>")
        issues: list[QualityIssue] = []

        for field_name in self.required_fields:
            if field_name not in record or record.get(field_name) in (None, ""):
                issues.append(QualityIssue(sample_id, field_name, "missing_required_field"))

        modality = record.get("modality")
        if modality is not None and modality not in self.valid_modalities:
            issues.append(QualityIssue(sample_id, "modality", f"unsupported_modality:{modality}"))

        annotation = record.get("annotation")
        if annotation is not None and not isinstance(annotation, dict):
            issues.append(QualityIssue(sample_id, "annotation", "annotation_must_be_object"))

        qa_status = record.get("qa_status", "needs_review")
        if qa_status not in VALID_QA_STATUS:
            issues.append(QualityIssue(sample_id, "qa_status", f"unsupported_qa_status:{qa_status}"))

        timestamp = record.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, (int, float)):
            issues.append(QualityIssue(sample_id, "timestamp", "timestamp_must_be_numeric_epoch"))

        return issues

    def evaluate(self, records: Iterable[dict[str, Any]]) -> QualityReport:
        materialized = list(records)
        report = QualityReport(total=len(materialized))

        for record in materialized:
            issues = self.validate_record(record)
            report.issues.extend(issues)
            if issues:
                report.invalid += 1
            else:
                report.valid += 1

            modality = record.get("modality") or "unknown"
            report.by_modality[modality] = report.by_modality.get(modality, 0) + 1

            if record.get("storage_uri"):
                report.stored += 1

            annotation = record.get("annotation")
            if isinstance(annotation, dict) and annotation:
                report.annotated += 1

            qa_status = record.get("qa_status", "needs_review")
            if qa_status == "pass":
                report.qa_passed += 1
            elif qa_status == "needs_review":
                report.needs_review += 1
            elif qa_status == "fail":
                report.rework_required += 1

        return report


def records_from_agv_snapshot(agvs: Iterable[dict[str, Any]], ts: float) -> list[dict[str, Any]]:
    """Convert live AGV telemetry into robotics learning-data QA records.

    The record shape is intentionally close to common dataset-governance concepts:
    sample identity, robot identity, modality, storage location, annotation, QA
    status, and metadata. It can later map to LeRobot-style episodes or a cloud
    data lake schema without changing the quality metrics.
    """
    records = []
    for agv in agvs:
        pred = agv.get("pred", "정상")
        conf = float(agv.get("conf", 0.0))
        qa_status = "pass" if conf >= 0.80 else "needs_review"
        if agv.get("status") == "warn" and conf < 0.60:
            qa_status = "fail"
        records.append({
            "sample_id": f"{agv.get('id', 'AGV')}-{int(ts * 1000)}",
            "robot_id": agv.get("id"),
            "timestamp": ts,
            "modality": "telemetry",
            "storage_uri": f"memory://telemetry/{agv.get('id')}/{int(ts * 1000)}",
            "annotation": {
                "label": pred,
                "category": "normal" if pred == "정상" else "fault",
                "confidence": conf,
            },
            "qa_status": qa_status,
            "metadata": {
                "floor": agv.get("floor"),
                "health": agv.get("health"),
                "trend_dir": agv.get("trend_dir"),
            },
        })
    return records

