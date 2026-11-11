"""Candidate lifetime and ranking checks on synthetic inputs, not field performance."""
import gc
import json
from pathlib import Path
import sys
import weakref

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diagnosis_evaluation import classification_metrics, select_candidate
import train_diagnosis

SERVING = ROOT / "data/processed"


@pytest.fixture
def candidate_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    metadata = json.loads((SERVING / "enhanced_meta.json").read_text())
    (data / "enhanced_meta.json").write_text(json.dumps(metadata))
    classes = len(metadata["err_map"])
    normal = metadata["err_map"]["정상"]
    per_asset = np.repeat(np.arange(classes), [40 if i == normal else 2 for i in range(classes)])
    labels = np.tile(per_asset, 6)
    rng = np.random.default_rng(7)
    for split in ("train", "val"):
        raw = rng.normal(size=(len(labels), 30, 7)).astype(np.float32)
        groups = np.repeat([f"{split}-{i}" for i in range(6)], len(per_asset))
        np.savez_compressed(data / f"enhanced_{split}.npz", X=raw,
                            S=np.zeros((len(labels), 9), dtype=np.float32), y=labels, groups=groups)
    return data


def test_native_search_releases_losers_and_saves_the_exact_selected_model(candidate_data, tmp_path, monkeypatch):
    before = (SERVING / "robot_pdm_enhanced.txt").read_bytes()
    original_classifier = train_diagnosis.lgb.LGBMClassifier
    original_event = train_diagnosis.record_event
    refs, live_before_creation, model_texts = [], [], {}

    def tracked_classifier(*args, **kwargs):
        gc.collect()
        live_before_creation.append(sum(ref() is not None for ref in refs))
        assert live_before_creation[-1] <= 1
        model = original_classifier(*args, **kwargs)
        refs.append(weakref.ref(model))
        return model

    def tracked_event(output, event, **fields):
        if event == "candidate_completed":
            current = refs[-1]()
            assert current is not None
            model_texts[fields["candidate_id"]] = current.booster_.model_to_string(
                num_iteration=fields["best_iteration"])
        original_event(output, event, **fields)

    monkeypatch.setattr(train_diagnosis.lgb, "LGBMClassifier", tracked_classifier)
    monkeypatch.setattr(train_diagnosis, "record_event", tracked_event)
    output = tmp_path / "candidate"
    report = train_diagnosis.train_candidates(candidate_data, output, SERVING, rounds=2,
                                               data_kind="synthetic_fixture")
    gc.collect()
    assert live_before_creation == [0] + [1] * 11
    assert len(refs) == len(report["experiments"]) == len(model_texts) == 12
    assert len(set(model_texts.values())) > 1
    assert all(ref() is None for ref in refs)
    assert all("model" not in row for row in report["experiments"])
    assert (output / "robot_pdm_enhanced.txt").read_text() == model_texts[report["selected_candidate"]]
    expected = select_candidate({row["candidate_id"]: row["selection_metrics"] for row in report["experiments"]})
    expected["dataset"] = "training_robot_holdout"
    assert report["selection_decision"] == expected
    retention = {"strategy": "selection_leader_only", "max_completed_models_between_candidates": 1,
                 "all_candidate_metrics_retained": True}
    assert report["candidate_model_retention"] == retention
    events = [json.loads(line) for line in (output / "training_events.jsonl").read_text().splitlines()]
    assert events[0]["candidate_model_retention"] == retention
    assert sum(row["event"] == "candidate_completed" for row in events) == 12
    assert sum(row["event"] == "candidate_selected" for row in events) == 1
    assert len(pd.read_csv(output / "selection_results.csv")) == 12
    assert json.loads((output / "experiment_report.json").read_text())["experiments"] == report["experiments"]
    assert not report["official_validation_used_for_selection"]
    assert not report["promotion"]["automatic_promotion"]
    assert "data_provenance_not_recorded_aihub" in report["promotion"]["reasons"]
    assert (SERVING / "robot_pdm_enhanced.txt").read_bytes() == before


def test_non_prefix_stable_policy_stops_before_official_comparison(candidate_data, tmp_path, monkeypatch):
    original_select = train_diagnosis.select_candidate
    original_metrics = train_diagnosis.classification_metrics
    metric_calls = []

    def unstable_policy(reports, limit):
        decision = original_select(reports, limit)
        ids = list(reports)
        decision["selected_candidate"] = ids[1] if len(ids) == 2 else ids[0]
        return decision

    def tracked_metrics(y, probabilities, names):
        metric_calls.append(len(y))
        return original_metrics(y, probabilities, names)

    def forbidden_comparison(*args):
        pytest.fail("a discarded model must never be replaced with an unrelated model in official comparison")

    monkeypatch.setattr(train_diagnosis, "select_candidate", unstable_policy)
    monkeypatch.setattr(train_diagnosis, "classification_metrics", tracked_metrics)
    monkeypatch.setattr(train_diagnosis, "load_booster", forbidden_comparison)
    output = tmp_path / "unstable-policy"
    with pytest.raises(ValueError, match="previously discarded candidate model"):
        train_diagnosis.train_candidates(candidate_data, output, SERVING, rounds=2,
                                          normal_ratios=(None, 2, 3), weight_modes=("unweighted",),
                                          data_kind="synthetic_fixture")
    events = [json.loads(line) for line in (output / "training_events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "selection_failed"
    assert "previously discarded candidate model" in events[-1]["detail"]
    completed = [row for row in events if row["event"] == "candidate_completed"]
    assert len(completed) == 3
    assert metric_calls == [row["selection_metrics"]["rows"] for row in completed]
    assert not any(row["event"] == "candidate_selected" for row in events)
    assert not (output / "experiment_report.json").exists()
    assert not (output / "robot_pdm_enhanced.txt").exists()


def fixture_metrics(counts):
    matrix = np.asarray(counts)
    y = np.repeat(np.arange(2), matrix.sum(axis=1))
    predicted = np.concatenate([np.repeat(np.arange(2), row) for row in matrix])
    return classification_metrics(y, np.eye(2)[predicted], ["정상", "fault"])


@pytest.mark.parametrize("order,expected", [("abcde", "d"), ("edcba", "e"), ("ab", "b"),
                                           ("bc", "c"), ("ce", "e")])
def test_fixed_ranking_never_resurrects_a_losing_prefix_candidate(order, expected):
    candidates = {
        "a": fixture_metrics([[900, 50], [20, 30]]),
        "b": fixture_metrics([[900, 50], [0, 50]]),
        "c": fixture_metrics([[940, 10], [40, 10]]),
        "d": fixture_metrics([[940, 10], [10, 40]]),
        "e": fixture_metrics([[940, 10], [10, 40]]),
    }
    seen, leader = {}, None
    for candidate_id in order:
        seen[candidate_id] = candidates[candidate_id]
        decision = select_candidate(seen)
        assert decision["selected_candidate"] in {leader, candidate_id}
        leader = decision["selected_candidate"]
    assert leader == expected
    assert leader == select_candidate({key: candidates[key] for key in order})["selected_candidate"]
