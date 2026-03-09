"""GoToAgent — high-level agent that picks goal cells as macro-actions."""

from __future__ import annotations

import numpy as np

from push_back.env.state import WorldState


class GoToAgent:
    """Uniformly samples from available goal indices each time it's free.

    Paired with a MacroResolver in the env, each chosen index maps to a
    goal (gx, gy) and the planner converts it to a primitive sequence.
    The agent only picks a new goal when the resolver is not busy.
    """

    def __init__(self, n_goals: int, rng: np.random.Generator | None = None) -> None:
        self.n_goals = n_goals
        self.rng = rng or np.random.default_rng()

    def act(self, state: WorldState, agent_id: int) -> int:
        return int(self.rng.integers(self.n_goals))
