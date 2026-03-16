"""RandomRobot — primitive movement robot with discrete kinematics.

Defines the Action enum and movement kinematics (FORWARD, TURN, STAY).
Picks a random primitive action each tick.
"""

from __future__ import annotations

from enum import IntEnum

import gymnasium
import numpy as np

from push_back.env.robots.base import BaseObservation
from push_back.env.robots.stand_still import StandStill
from push_back.env.state import BallColor, GRID_SIZE, HEADING_DELTAS, Pose
from push_back.env.collision import is_cell_blocked

DEFAULT_TURN_TICKS: int = 3


class Action(IntEnum):
    """Discrete robot actions."""

    STAY = 0
    FORWARD = 1
    TURN_LEFT = 2  # counterclockwise (+1 heading index)
    TURN_RIGHT = 3  # clockwise (-1 heading index)


# HEADING_DELTAS imported from state.py


class RandomRobot(StandStill):
    """Uniformly samples from the 4 primitive actions each tick."""

    def __init__(
        self,
        turn_ticks: int = DEFAULT_TURN_TICKS,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.turn_ticks = turn_ticks
        self._turn_progress: int = 0
        self._turn_direction: int = 0
        self.rng = rng or np.random.default_rng()

    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(len(Action))

    def valid_actions(self, pose: Pose) -> list[int]:
        """Return list of actions that won't move into a blocked cell."""
        valid = [Action.STAY, Action.TURN_LEFT, Action.TURN_RIGHT]
        dx, dy = HEADING_DELTAS[pose.heading]
        nx, ny = pose.x + dx, pose.y + dy
        blocked = getattr(self, "_blocked_cells", set())
        if not is_cell_blocked(nx, ny, blocked):
            valid.append(Action.FORWARD)
        return valid

    def tick(self, obs: BaseObservation, agent_id: int) -> int:
        valid = self.valid_actions(
            Pose(
                int(obs.agents[agent_id][0]),
                int(obs.agents[agent_id][1]),
                int(obs.agents[agent_id][2]),
            )
        )
        return int(self.rng.choice(valid))

    def resolve(self, action: int, pose: Pose) -> int:
        return action

    def apply(self, action: int, pose: Pose) -> Pose:
        primitive = self.resolve(action, pose)
        new = Pose(pose.x, pose.y, pose.heading)

        if primitive == Action.FORWARD:
            dx, dy = HEADING_DELTAS[pose.heading]
            new.x, new.y = pose.x + dx, pose.y + dy
            self._turn_progress = 0
            self._turn_direction = 0
            return new

        if primitive in (Action.TURN_LEFT, Action.TURN_RIGHT):
            direction = 1 if primitive == Action.TURN_LEFT else -1
            if direction == self._turn_direction:
                self._turn_progress += 1
            else:
                self._turn_progress = 1
                self._turn_direction = direction
            if self._turn_progress >= self.turn_ticks:
                new.heading = (pose.heading + direction) % 8
                self._turn_progress = 0
                self._turn_direction = 0
            return new

        # STAY
        self._turn_progress = 0
        self._turn_direction = 0
        return new

    def filter_balls(
        self, balls: list[BallColor], own_color: BallColor
    ) -> tuple[list[BallColor], list[BallColor]]:
        """Keep own-color balls, spit out opponent balls."""
        keep = [b for b in balls if b == own_color]
        spit = [b for b in balls if b != own_color]
        return keep, spit

    def reset(self) -> None:
        self._turn_progress = 0
        self._turn_direction = 0
