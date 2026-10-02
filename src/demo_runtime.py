"""Explicit, deterministic exhibition controls; never model evaluation data."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class DemoControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paused: bool | None = None
    speed: Literal[0.25, 0.5, 1.0] | None = None
    asset_id: str | None = Field(default=None, max_length=80)
    scenario: Literal["replay", "normal", "watch", "degrading", "battery", "network"] | None = None


SCENARIOS = {
    "normal": {"pred": "정상", "health": 98, "trend": [0] * 12,
               "sensors": {"vib": 1.8, "batt": 86.0, "temp": 38.0}},
    "watch": {"pred": "정상", "health": 68, "trend": [0] * 12,
              "sensors": {"vib": 4.6, "batt": 42.0, "temp": 51.0}},
    "degrading": {"pred": "정상", "health": 48, "trend": [0] * 6 + [1] * 5 + [0],
                  "sensors": {"vib": 5.8, "batt": 28.0, "temp": 59.0}},
    "battery": {"pred": "E-RBT-B", "health": 22, "trend": [1] * 6 + [3] * 6,
                "sensors": {"vib": 4.2, "batt": 16.0, "temp": 64.0}},
    "network": {"pred": "E-RBT-N", "health": 28, "trend": [0] * 6 + [3] * 6,
                "sensors": {"vib": 0.0, "batt": 72.0, "temp": 40.0}},
}


class DemoRuntime:
    def __init__(self):
        self.paused = False
        self.speed = 0.5
        self.overrides: dict[str, str] = {}

    def state(self) -> dict:
        return {"paused": self.paused, "speed": self.speed,
                "overrides": dict(self.overrides), "scope": "shared_server_session"}

    def update(self, request: DemoControl, asset_ids: set[str]) -> dict:
        if request.scenario is not None and request.asset_id not in asset_ids:
            raise ValueError("unknown_asset")
        if request.asset_id is not None and request.scenario is None:
            raise ValueError("scenario_required")
        if request.paused is not None:
            self.paused = request.paused
        if request.speed is not None:
            self.speed = request.speed
        if request.scenario is not None:
            if request.scenario == "replay":
                self.overrides.pop(request.asset_id, None)
            else:
                self.overrides[request.asset_id] = request.scenario
        return self.state()

    def sample(self, asset_id: str) -> dict | None:
        name = self.overrides.get(asset_id)
        return {**SCENARIOS[name], "name": name} if name else None
