"""Macro-F1 iteration selection on synthetic predictions, not field accuracy."""
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnosis_evaluation import macro_f1_eval


def test_metric_matches_unweighted_macro_f1_and_maximizes():
    y = np.array([0] * 90 + [1] * 5 + [2] * 5, dtype=np.float32)
    probabilities = np.zeros((len(y), 3))
    probabilities[:, 0] = 1
    name, score, higher_is_better = macro_f1_eval(y, probabilities)
    assert name == "macro_f1"
    assert higher_is_better is True
    assert score == pytest.approx(f1_score(y, probabilities.argmax(1), labels=[0, 1, 2], average="macro"))
    assert score < .4


def test_missing_class_still_counts_in_macro_denominator():
    assert macro_f1_eval([0, 1], [[1, 0, 0], [0, 1, 0]]) == ("macro_f1", pytest.approx(2 / 3), True)


def test_perfect_predictions_score_one():
    assert macro_f1_eval(np.arange(3), np.eye(3)) == ("macro_f1", 1.0, True)


@pytest.mark.parametrize("labels,predictions", [
    ([], np.zeros((0, 3))), ([0, 1], [0.1, 0.9]),
    ([0, 1], np.ones((1, 3))), ([0], np.ones((1, 1))),
    ([[0]], np.ones((1, 3))), ([0.5], np.ones((1, 3))),
    ([-1], np.ones((1, 3))), ([3], np.ones((1, 3))),
    ([np.nan], np.ones((1, 3))), ([0], [[np.nan, 0, 1]]),
    ([0], [[0, np.inf, 1]]),
])
def test_metric_rejects_invalid_labels_or_predictions(labels, predictions):
    with pytest.raises(ValueError, match="macro-F1"):
        macro_f1_eval(labels, predictions)


def test_real_lightgbm_stops_on_f1_plateau_without_builtin_logloss():
    labels = np.repeat(np.arange(3), 40)
    features = pd.DataFrame({"signal": labels.astype(float), "offset": np.tile(np.arange(40), 3)})
    held_labels = np.repeat(np.arange(3), 20)
    held_features = pd.DataFrame({"signal": held_labels.astype(float),
                                  "offset": np.tile(np.linspace(.25, 39.25, 20), 3)})
    model = lgb.LGBMClassifier(objective="multiclass", metric="None", n_estimators=20,
                               min_child_samples=1, num_leaves=3, random_state=42, n_jobs=1,
                               verbosity=-1, deterministic=True, force_col_wise=True)
    history = {}
    model.fit(features, labels, eval_set=[(held_features, held_labels)], eval_names=["robot_holdout"],
              eval_metric=macro_f1_eval,
              callbacks=[lgb.record_evaluation(history), lgb.early_stopping(3, first_metric_only=True, verbose=False)])
    assert set(history["robot_holdout"]) == {"macro_f1"}
    scores = history["robot_holdout"]["macro_f1"]
    assert scores == [1.0] * 4
    assert model.best_iteration_ == 1
    assert model.best_score_["robot_holdout"]["macro_f1"] == 1.0
    assert model.booster_.params["objective"] == "multiclass"
