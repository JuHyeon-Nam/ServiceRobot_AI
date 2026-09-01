import json
import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from build_rul_dataset import build_rul_rows, read_events_csv, read_failure_labels, write_csv
from train_rul_baseline import load_rul_training_csv, save_artifacts, train_rul_baseline

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _training_csv(tmp_path):
    events = read_events_csv(os.path.join(FIXTURES, "rul_events.csv"))
    labels = read_failure_labels(os.path.join(FIXTURES, "rul_failure_labels.csv"))
    rows = build_rul_rows(events, labels)
    path = tmp_path / "rul_training.csv"
    write_csv(rows, str(path))
    return path


def test_load_rul_training_csv_uses_observed_failures_only(tmp_path):
    path = _training_csv(tmp_path)

    x, y, rows = load_rul_training_csv(str(path))

    assert x.shape == (12, 8)
    assert y.shape == (12,)
    assert len(rows) == 12
    assert all(str(row["event_observed"]) == "1" for row in rows)


def test_train_rul_baseline_writes_model_and_metadata(tmp_path):
    path = _training_csv(tmp_path)
    model_path = tmp_path / "rul_baseline.pkl"
    meta_path = tmp_path / "rul_baseline_meta.json"

    result = train_rul_baseline(str(path), random_state=7)
    metadata = save_artifacts(result, str(model_path), str(meta_path))

    assert result["schema"] == "fab.rul.baseline_model.v1"
    assert result["contract_schema"] == "fab.rul.calibration.v1"
    assert result["model_kind"] == "gradient_boosting_regressor"
    assert result["rows"] == 12
    assert result["test_rows"] > 0
    assert "mae_min" in result["metrics"]
    assert "median_baseline_mae_min" in result["metrics"]
    assert model_path.exists() and meta_path.exists()
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved == metadata
    assert saved["feature_fields"][0] == "health"
