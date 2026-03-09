"""Agent interface and example implementations.

Each agent receives the full WorldState and its own index (0-3),
and returns a discrete Action (int). Agents 0-1 are "red", 2-3 are "blue".
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from push_back.env.state import Action, WorldState


class Agent(Protocol):
    """Minimal interface any agent must satisfy."""

    def act(self, state: WorldState, agent_id: int) -> int: ...


# --------------- concrete agents ---------------


class StandStill:
    """Does nothing. Useful as a placeholder."""

    def act(self, state: WorldState, agent_id: int) -> int:
        return Action.STAY


class RandomAgent:
    """Picks a random action each tick."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, state: WorldState, agent_id: int) -> int:
        return int(self.rng.integers(len(Action)))
