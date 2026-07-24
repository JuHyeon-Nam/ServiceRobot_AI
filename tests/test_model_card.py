import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from model_card import build_model_card


def test_model_card_artifact_and_feature_contract():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    card = build_model_card(data_dir)

    assert card["model_id"] == "robot-pdm-lightgbm-enhanced"
    assert card["artifact"]["path"] == "data/processed/robot_pdm_enhanced.txt"
    assert card["artifact"]["size_bytes"] > 1_000_000
    assert len(card["artifact"]["sha256"]) == 64
    assert card["feature_engineering"]["contract_ok"] is True
    assert card["feature_engineering"]["n_features"] == 249
    assert card["input_contract"]["window_timesteps"] == 30
    assert card["input_contract"]["excluded_dynamic_sensors"] == ["x", "y"]
    assert "distance" in card["input_contract"]["static_context"]
    assert card["performance"]["official_validation"]["accuracy"] == 0.9329
    assert "/api/drift" in card["operations"]["observability"]
