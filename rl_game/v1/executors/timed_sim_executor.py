from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Optional

from actions import ActionType, MacroAction
from config_v1 import (
    ACTIVE_AGENTS,
    COLLECT_TIME,
    DRIVE_SPEED_CELLS_PER_SECOND,
    SCORE_TIME,
    TOGGLE_TIME,
    WAIT_TIME,
)
from executors.base_executor import ActionExecutor


@dataclass
class PendingAction:
    action: MacroAction
    start_time: float
    end_time: float
    payload: Dict[str, Any]


class TimedSimExecutor(ActionExecutor):
    """
    V1 macro-action timing model.

    It does not simulate wheel-by-wheel motion. Instead it estimates
    travel time from the full GameEnv coordinates, waits until the
    macro action finishes, and lets GameEnvBridge apply the completion.
    """

    def __init__(self):
        self.pending: Dict[str, Optional[PendingAction]] = {}
        self.reset()

    def reset(self):
        self.pending = {agent_id: None for agent_id in ACTIVE_AGENTS}

    def _travel_time(self, start, target) -> float:
        distance = sqrt((target[0] - start[0]) ** 2 + (target[1] - start[1]) ** 2)
        return distance / DRIVE_SPEED_CELLS_PER_SECOND

    def start_action(self, agent_id, action, bridge, current_time: float):
        if self.is_busy(agent_id):
            raise ValueError(f"{agent_id} is already busy")

        start_position = bridge.agent_position(agent_id)
        payload: Dict[str, Any] = {}

        if action.action_type == ActionType.WAIT:
            duration = WAIT_TIME

        elif action.action_type == ActionType.COLLECT:
            obj = bridge.nearest_object_in_region(agent_id, action.target)
            if obj is None:
                raise ValueError(f"No available object in {action.target}")

            target_position = (obj["row"], obj["col"])
            duration = self._travel_time(start_position, target_position) + COLLECT_TIME
            payload = {
                "object_id": obj["object_id"],
                "channel": obj["channel"],
                "row": obj["row"],
                "col": obj["col"],
            }

        elif action.action_type == ActionType.SCORE:
            target_position = bridge.goal_position(action.target)
            duration = self._travel_time(start_position, target_position) + SCORE_TIME
            payload = {
                "goal_id": action.target,
                "row": target_position[0],
                "col": target_position[1],
            }

        elif action.action_type == ActionType.TOGGLE:
            targets = bridge.toggle_targets()
            if action.target not in targets:
                raise ValueError(f"Unknown toggle target: {action.target}")
            target_position = targets[action.target]
            duration = self._travel_time(start_position, target_position) + TOGGLE_TIME
            payload = {
                "region": action.target,
                "row": target_position[0],
                "col": target_position[1],
            }

        else:
            raise ValueError(f"Unsupported macro action: {action}")

        pending = PendingAction(
            action=action,
            start_time=float(current_time),
            end_time=float(current_time + duration),
            payload=payload,
        )
        self.pending[agent_id] = pending
        return pending

    def advance_to_next_event(self, current_time: float, episode_time: float):
        active = {
            agent_id: pending
            for agent_id, pending in self.pending.items()
            if pending is not None
        }

        if not active:
            raise RuntimeError("No active macro actions")

        next_event_time = min(pending.end_time for pending in active.values())
        next_time = min(float(next_event_time), float(episode_time))

        completed: Dict[str, PendingAction] = {}

        for agent_id, pending in active.items():
            if pending.end_time <= next_time + 1e-9:
                completed[agent_id] = pending
                self.pending[agent_id] = None

        return next_time, completed

    def is_busy(self, agent_id: str) -> bool:
        return self.pending.get(agent_id) is not None

    def get_remaining_time(self, agent_id: str, current_time: float) -> float:
        pending = self.pending.get(agent_id)
        if pending is None:
            return 0.0
        return max(0.0, float(pending.end_time - current_time))

    def get_current_action(self, agent_id: str):
        pending = self.pending.get(agent_id)
        return pending.action if pending is not None else None
