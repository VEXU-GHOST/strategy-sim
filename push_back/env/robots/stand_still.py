"""StandStill — concrete base class for all robots.

Provides default observation space and a trivial single-action interface.
Subclasses override what they need.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from push_back.env.robots.base import BaseObservation, BaseRobot
from push_back.env.state import GRID_SIZE, Pose


class StandStill(BaseRobot):
    """Default robot: single action (stay), no movement."""

    def observation_space(self, n_balls: int) -> gymnasium.spaces.Dict:
        return gymnasium.spaces.Dict(
            {
                "balls": gymnasium.spaces.Box(
                    0, GRID_SIZE - 1, shape=(n_balls, 3), dtype=np.int32
                ),
                "agents": gymnasium.spaces.Box(
                    low=np.array([[0, 0, 0]] * 4, dtype=np.int32),
                    high=np.array(
                        [[GRID_SIZE - 1, GRID_SIZE - 1, 7]] * 4, dtype=np.int32
                    ),
                    dtype=np.int32,
                ),
                "score": gymnasium.spaces.Box(0, np.inf, shape=(2,), dtype=np.int32),
            }
        )

    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(1)

    def tick(self, obs: BaseObservation, agent_id: int) -> int:
        return 0

    def resolve(self, action: int, pose: Pose) -> int:
        return 0

    def apply(self, action: int, pose: Pose) -> Pose:
        return Pose(pose.x, pose.y, pose.heading)
