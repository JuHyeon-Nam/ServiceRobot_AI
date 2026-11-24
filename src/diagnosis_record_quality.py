"""Raw-field checks for explicit preprocessing experiments; never impute labels or values."""
import math

from audit_diagnosis_source import NUMERIC_PATHS
from build_enhanced_dataset import CROWD, DYN, STAT_NUM

MISSING = object()
FLOAT32_MAX = 3.4028234663852886e38


def numeric_issue(value):
    if value is MISSING:
        return "missing_defaults_to_zero"
    if value is None:
        return "null_defaults_to_zero"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "invalid_numeric_defaults_to_zero"
    except OverflowError:
        return "outside_float32"
    if not math.isfinite(number):
        return "nonfinite"
    if abs(number) > FLOAT32_MAX:
        return "outside_float32"
    return None


def record_quality(document, record, metadata):
    issues = {}
    for name, path in NUMERIC_PATHS.items():
        value = document
        for part in path:
            value = value.get(part, MISSING) if isinstance(value, dict) else MISSING
        reason = numeric_issue(value)
        if reason:
            issues[name] = reason
    dynamic = {name: issues[name] for name in DYN if name in issues}
    context = {name: issues[name] for name in STAT_NUM if name in issues}
    site = document.get("siteData", {})
    crowd = site.get("crowd", MISSING) if isinstance(site, dict) else MISSING
    if not isinstance(crowd, str) or crowd not in CROWD:
        context["crowd"] = "missing_or_unknown_defaults_to_middle"
    for field, mapping in (("deviceType", "devtype_map"), ("mainState", "mainstate_map")):
        value = record.get(field)
        if not isinstance(value, str) or value not in metadata[mapping]:
            context[field] = "missing_or_unknown_frozen_category"
    return {"dynamic_issues": dynamic, "context_issues": context,
            "dynamic_usable": not dynamic, "target_context_usable": not context}
