"""Agent protocol and ActionResolver for macro-action support.

Two agent types plug into the env:
  - **Low-level**: returns a primitive Action int each tick (STAY/FORWARD/TURN_*).
  - **High-level**: returns a macro-action index; the env's ActionResolver expands
    it into a sequence of primitives via the planner and drains one per tick.

# TODO: consider Option B — let high-level agents issue a new goal *or*
# CONTINUE each tick, enabling mid-move cancellation and replanning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Protocol

from push_back.env.state import Action, WorldState, Pose


class Agent(Protocol):
    """An agent returns an int action each tick."""

    def act(self, state: WorldState, agent_id: int) -> int: ...


class ActionResolver(ABC):
    """Translates an agent's action int into a per-tick primitive.

    Low-level agents use ``PrimitiveResolver`` (identity).
    High-level agents use ``MacroResolver`` which plans a path and queues it.
    """

    @abstractmethod
    def resolve(self, action: int, pose: Pose) -> int:
        """Return the primitive Action to execute this tick."""
        ...

    @abstractmethod
    def busy(self) -> bool:
        """True if the resolver is still executing a queued macro-action."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear any queued actions (called on env reset)."""
        ...


class PrimitiveResolver(ActionResolver):
    """Identity resolver for agents that directly emit primitives."""

    def resolve(self, action: int, pose: Pose) -> int:
        return action

    def busy(self) -> bool:
        return False

    def reset(self) -> None:
        pass


class MacroResolver(ActionResolver):
    """Resolver that converts goal-cell macro-actions into primitive queues.

    *goals* is a list of (gx, gy) grid cells.  When the agent picks action
    index *i*, the planner generates a primitive sequence from the current pose
    to goals[i].  The resolver drains one primitive per tick until the queue is
    empty.  While busy, the agent's action is **ignored** (options framework).
    """

    def __init__(self, goals: list[tuple[int, int]], turn_ticks: int = 3) -> None:
        self.goals = goals
        self.turn_ticks = turn_ticks
        self._queue: deque[int] = deque()

    def resolve(self, action: int, pose: Pose) -> int:
        from push_back.env.planner import plan  # deferred to avoid circular import

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
        self._queue.clear()
