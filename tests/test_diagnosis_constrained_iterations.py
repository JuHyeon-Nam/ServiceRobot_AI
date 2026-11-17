"""Constrained iteration selection on synthetic/scripted predictions, not real-data gains."""
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnosis_evaluation import macro_f1_eval, make_constrained_macro_f1_eval


def metrics(y, predicted, limit=.05, normal=0, classes=3):
    return make_constrained_macro_f1_eval(normal, limit)(y, np.eye(classes)[predicted])


def test_feasible_round_beats_a_higher_macro_f1_infeasible_round():
    y = np.repeat([0, 1, 2], [20, 4, 4])
    feasible = np.zeros(len(y), dtype=int)
    infeasible = y.copy()
    infeasible[:4] = 1
    before, after = metrics(y, feasible), metrics(y, infeasible)
    assert [row[0] for row in before] == ["constrained_macro_f1", "macro_f1", "normal_false_alarm_rate"]
    assert [row[2] for row in before] == [True, True, False]
    assert before[1][1] < after[1][1]
    assert before[0][1] > after[0][1]
    assert before[0][1] == before[1][1]
    assert after[0][1] == after[1][1] - 2
    assert before[2][1] == 0 and after[2][1] == .2


def test_feasible_rounds_are_ranked_by_plain_unweighted_macro_f1():
    y = np.repeat([0, 1, 2], [20, 4, 4])
    normal_only, perfect = metrics(y, np.zeros(len(y), dtype=int)), metrics(y, y)
    assert perfect[0] == ("constrained_macro_f1", 1., True)
    assert perfect[0][1] > normal_only[0][1]
    assert perfect[2][1] == normal_only[2][1] == 0


def test_no_feasible_round_preserves_an_analysis_only_macro_f1_ranking():
    y = np.repeat([0, 1, 2], [20, 4, 4])
    good, worse = y.copy(), np.zeros(len(y), dtype=int)
    good[:4] = worse[:4] = 1
    first, second = metrics(y, good), metrics(y, worse)
    assert first[0][1] < 0 and second[0][1] < 0
    assert first[1][1] > second[1][1] and first[0][1] > second[0][1]


@pytest.mark.parametrize("limit,feasible", [(1 / 3, False), (.333333, False), (.34, True)])
def test_constraint_uses_exact_counts_not_rounded_or_float_rate_comparison(limit, feasible):
    values = metrics([0, 0, 0, 1, 2], [1, 0, 0, 1, 2], limit)
    assert values[2][1] == 1 / 3
    assert (values[0][1] >= 0) == feasible


@pytest.mark.parametrize("normal", [0, 1, 2])
def test_normal_class_position_and_inputs_are_preserved(normal):
    y = np.tile(np.arange(3), 3)
    predicted = y.copy()
    predicted[np.flatnonzero(y == normal)[0]] = (normal + 1) % 3
    probabilities = np.eye(3)[predicted]
    before_y, before_probabilities = y.copy(), probabilities.copy()
    values = make_constrained_macro_f1_eval(normal, .34)(y, probabilities)
    assert values[0][1] == values[1][1] == macro_f1_eval(y, probabilities)[1]
    assert values[2][1] == 1 / 3
    np.testing.assert_array_equal(y, before_y)
    np.testing.assert_array_equal(probabilities, before_probabilities)


@pytest.mark.parametrize("normal", [-1, 1.5, True])
def test_invalid_normal_index_fails_at_configuration(normal):
    with pytest.raises(ValueError, match="normal class"):
        make_constrained_macro_f1_eval(normal)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), -1, 1.1, True, ".05"])
def test_invalid_limit_fails_at_configuration(limit):
    with pytest.raises(ValueError, match="false-alarm limit"):
        make_constrained_macro_f1_eval(0, limit)


def test_absent_normal_and_wrong_class_order_cannot_be_treated_as_feasible():
    with pytest.raises(ValueError, match="actual normal"):
        metrics([1, 2], [1, 2])
    with pytest.raises(ValueError, match="outside the prediction"):
        metrics([0, 1, 2], [0, 1, 2], normal=3)


def test_invalid_predictions_still_use_the_common_macro_f1_input_checks():
    with pytest.raises(ValueError, match="macro-F1"):
        make_constrained_macro_f1_eval(0)([0, 1], [[np.nan, 0, 1], [0, 1, 0]])


@pytest.mark.parametrize("mode,expected_best", [("macro_f1", 2), ("constrained_macro_f1", 1)])
def test_native_early_stopping_respects_feasibility_on_scripted_predictions(mode, expected_best):
    y = np.repeat([0, 1, 2], [20, 4, 4])
    features = pd.DataFrame({"signal": y.astype(float), "offset": np.arange(len(y))})
    held_features = features + .1
    infeasible = y.copy()
    infeasible[:4] = 1
    evaluate = macro_f1_eval if mode == "macro_f1" else make_constrained_macro_f1_eval(0)
    calls = []

    def scripted_evaluation(labels, ignored_model_predictions):
        predicted = np.zeros(len(y), dtype=int) if not calls else infeasible
        calls.append(len(labels))
        return evaluate(labels, np.eye(3)[predicted])

    model = lgb.LGBMClassifier(objective="multiclass", metric="None", n_estimators=10,
                               min_child_samples=1, num_leaves=3, random_state=42, n_jobs=1,
                               verbosity=-1, deterministic=True, force_col_wise=True)
    history = {}
    model.fit(features, y, eval_set=[(held_features, y)], eval_names=["robot_holdout"],
              eval_metric=scripted_evaluation,
              callbacks=[lgb.record_evaluation(history), lgb.early_stopping(3, first_metric_only=True, verbose=False)])
    assert model.best_iteration_ == expected_best
    assert model.booster_.params["objective"] == "multiclass"
    values = history["robot_holdout"][mode]
    assert values[expected_best - 1] == max(values)
    assert len(calls) == expected_best + 3
    if mode == "constrained_macro_f1":
        assert history["robot_holdout"]["macro_f1"][0] < history["robot_holdout"]["macro_f1"][1]
        assert values[0] >= 0 and all(value < 0 for value in values[1:])
        assert history["robot_holdout"]["normal_false_alarm_rate"] == [0, .2, .2, .2]
