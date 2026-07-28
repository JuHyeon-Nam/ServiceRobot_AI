import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from reviewer_brief import build_reviewer_brief


def test_reviewer_brief_contract():
    brief = build_reviewer_brief()

    assert brief["project"] == "ServiceRobot_AI"
    assert brief["review_time_minutes"] == 3
    assert brief["start_here"][0]["path"] == "/twin"
    assert len(brief["proof_points"]) >= 5
    assert any(item["role"] == "Data / AI Engineer" for item in brief["role_mapping"])
    assert any("/api/model-card" in item["evidence"] for item in brief["role_mapping"])
    assert "portfolio_demo" in brief["current_status"]
