"""RandomAgent — picks a random primitive action each tick."""

from __future__ import annotations

import numpy as np

from push_back.env.state import Action, WorldState


class RandomAgent:
    """Uniformly samples from the 4 primitive actions each tick."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, state: WorldState, agent_id: int) -> int:
        return int(self.rng.integers(len(Action)))
