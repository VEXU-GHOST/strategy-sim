"""Base classes for robot types, observations, and action spaces.

Each robot type defines what it can observe, what actions it supports,
and how to resolve high-level actions into simulation primitives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium
import numpy as np

from push_back.env.state import WorldState, Pose


@dataclass(frozen=True)
class BaseObservation:
    """Typed observation container.

    Subclasses may add fields; the base holds the common full-field view.
    """

    balls: np.ndarray    # (N, 3) int32 grid positions + color
    agents: np.ndarray   # (4, 3) int32 [x, y, heading]
    score: np.ndarray    # (2,) int32


@dataclass(frozen=True)
class BaseAction:
    """Wraps a gymnasium action space definition for a robot type."""

    space: gymnasium.Space


class BaseRobot(ABC):
    """Abstract robot type.

    Owns its observation space, action space, and action resolver.
    Concrete subclasses implement ``tick`` (the decision function) and
    define what gymnasium spaces they operate over.
    """

    @abstractmethod
    def observation_space(self, n_balls: int) -> gymnasium.Space:
        """Return the gymnasium observation space for this robot type."""
        ...

    @abstractmethod
    def action_space(self) -> gymnasium.Space:
        """Return the gymnasium action space for this robot type."""
        ...

    @abstractmethod
    def tick(self, obs: BaseObservation, agent_id: int) -> int:
        """Choose an action given an observation. Returns an action int."""
        ...

    @abstractmethod
    def resolve(self, action: int, pose: Pose) -> int:
        """Convert this robot's action int into a simulation primitive."""
        ...

    @abstractmethod
    def apply(self, action: int, pose: Pose) -> Pose:
        """Apply an action to a pose and return the new pose.

        Handles resolve + any internal kinematics (e.g. turn tracking).
        """
        ...

    def reset(self) -> None:
        """Called on env reset. Override to clear internal state."""

    def set_blocked_cells(self, blocked: set[tuple[int, int]]) -> None:
        """Store blocked cells for action masking. Called by env after reset."""
        self._blocked_cells: set[tuple[int, int]] = blocked

    def busy(self) -> bool:
        """True if the robot is mid-macro-action. Override for macro robots."""
        return False

    @staticmethod
    def build_obs(state: WorldState) -> BaseObservation:
        """Build a BaseObservation from the current WorldState."""
        agent_arr = np.array(
            [[p.x, p.y, p.heading] for p in state.agents], dtype=np.int32
        )
        return BaseObservation(
            balls=state.balls_on_field.copy(),
            agents=agent_arr,
            score=np.array(state.score, dtype=np.int32),
        )
