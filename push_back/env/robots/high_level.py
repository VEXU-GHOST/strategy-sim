"""GoToRobot — high-level robot that navigates to goal cells."""

from __future__ import annotations

from collections import deque

import gymnasium
import numpy as np

from push_back.env.robots.base import BaseObservation
from push_back.env.robots.random_robot import Action, RandomRobot
from push_back.env.field import DEFAULT_NAV_GOALS
from push_back.env.state import Pose


class GoToRobot(RandomRobot):
    """Picks a goal cell; the planner converts it to primitives.

    *goals* is a list of (gx, gy) grid cells. Action index *i* means
    "navigate to goals[i]".  The robot queues planned primitives and drains
    one per tick. While busy, the chosen action is **ignored** (options
    framework).

    # TODO: consider Option B — let high-level robots issue a new goal *or*
    # CONTINUE each tick, enabling mid-move cancellation and replanning.
    """

    def __init__(
        self,
        goals: list[tuple[int, int]] | None = None,
        turn_ticks: int = 3,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(turn_ticks=turn_ticks)
        self.goals = goals if goals is not None else DEFAULT_NAV_GOALS
        self.rng = rng or np.random.default_rng()
        self._queue: deque[int] = deque()

    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(len(self.goals))

    def tick(self, obs: BaseObservation, agent_id: int) -> int:
        return int(self.rng.integers(len(self.goals)))

    def resolve(self, action: int, pose: Pose) -> int:
        from push_back.env.planner import plan

        if self._queue:
            return self._queue.popleft()

        if 0 <= action < len(self.goals):
            gx, gy = self.goals[action]
            self._queue = plan(pose, gx, gy, self.turn_ticks)
            if self._queue:
                return self._queue.popleft()

        return Action.STAY

    def busy(self) -> bool:
        return len(self._queue) > 0

    def reset(self) -> None:
        super().reset()
        self._queue.clear()
