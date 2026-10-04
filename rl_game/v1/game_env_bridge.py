from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from actions import ActionType, MacroAction
from config_v1 import REGIONS, TEAM_A_SCORE_INDEX


class GameEnvBridge:
    """
    Thin compatibility layer around the teammate's full GameEnv.

    Only this file should know about GameEnv internals such as
    `_agents`, `_board`, `goalStates`, and `zoneColors`.

    If GameEnv changes later, update this bridge instead of the
    RL policy, StrategyEnv, or state/action definitions.
    """

    def __init__(self, game_env: Any):
        self.env = game_env
        self.last_raw_state: Optional[dict] = None

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def reset(self) -> dict:
        self.last_raw_state = self.env.reset()
        return self.last_raw_state

    # ---------------------------------------------------------
    # Basic simulator information
    # ---------------------------------------------------------

    @property
    def board_size(self) -> int:
        if hasattr(self.env, "_board_size"):
            return int(self.env._board_size)
        if hasattr(self.env, "_board"):
            return int(np.asarray(self.env._board).shape[-1])
        raise AttributeError("GameEnv does not expose board size")

    def region_of(self, row: float, col: float) -> str:
        half = self.board_size / 2.0

        if row < half and col < half:
            return "q0"
        if row < half and col >= half:
            return "q1"
        if row >= half and col < half:
            return "q2"
        return "q3"

    # ---------------------------------------------------------
    # Agents
    # ---------------------------------------------------------

    def agent_position(self, agent_id: str) -> Tuple[int, int]:
        agent = self.env._agents[agent_id]
        return int(agent.row), int(agent.col)

    def held_item_channels(self, agent_id: str) -> List[int]:
        held = np.asarray(self.env._agents[agent_id].held_items)
        return [int(i) for i, value in enumerate(held) if value > 0]

    def is_carrying(self, agent_id: str) -> bool:
        return len(self.held_item_channels(agent_id)) > 0

    # ---------------------------------------------------------
    # Field state
    # ---------------------------------------------------------

    def object_states(self) -> List[Dict[str, Any]]:
        """
        Read objects directly from GameEnv's board tensor.

        We deliberately keep object types as channel indices for V1
        because the current GameEnv channel semantics are still being
        finalized. This avoids hard-coding an incorrect semantic mapping.
        """
        board = np.asarray(self.env._board)
        objects: List[Dict[str, Any]] = []

        if board.ndim != 3:
            return objects

        for channel, row, col in np.argwhere(board > 0):
            count = int(board[channel, row, col])
            objects.append(
                {
                    "object_id": f"obj_c{int(channel)}_r{int(row)}_c{int(col)}",
                    "channel": int(channel),
                    "row": int(row),
                    "col": int(col),
                    "region": self.region_of(int(row), int(col)),
                    "count": count,
                }
            )

        return objects

    def goal_states(self):
        value = getattr(self.env, "goalStates", [])
        return np.asarray(value).copy().tolist()

    def toggle_states(self):
        value = getattr(self.env, "zoneColors", [])
        return np.asarray(value).copy().tolist()

    def loader_states(self):
        red = np.asarray(getattr(self.env, "redMatchload", [])).copy().tolist()
        blue = np.asarray(getattr(self.env, "blueMatchload", [])).copy().tolist()
        return {"red": red, "blue": blue}

    # ---------------------------------------------------------
    # Strategic targets
    # ---------------------------------------------------------

    def collect_targets(self) -> List[str]:
        regions = {obj["region"] for obj in self.object_states() if obj["count"] > 0}
        return [region for region in REGIONS if region in regions]

    def nearest_object_in_region(
        self,
        agent_id: str,
        region: str,
    ) -> Optional[Dict[str, Any]]:
        row, col = self.agent_position(agent_id)
        candidates = [
            obj
            for obj in self.object_states()
            if obj["region"] == region and obj["count"] > 0
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda obj: (obj["row"] - row) ** 2 + (obj["col"] - col) ** 2,
        )

    def goal_targets(self) -> Dict[str, Tuple[int, int]]:
        field_objects = getattr(self.env, "fieldObject", None)
        if field_objects is None:
            return {}

        result: Dict[str, Tuple[int, int]] = {}
        for i, item in enumerate(np.asarray(field_objects)):
            if len(item) < 2:
                continue
            result[f"goal_{i}"] = (int(item[0]), int(item[1]))
        return result

    def goal_position(self, goal_id: str) -> Tuple[int, int]:
        goals = self.goal_targets()
        if goal_id not in goals:
            raise ValueError(f"Unknown goal target: {goal_id}")
        return goals[goal_id]

    def toggle_targets(self) -> Dict[str, Tuple[int, int]]:
        # Exact toggle coordinates are not exposed reliably by the current
        # GameEnv yet. Keep the action type in the interface, but do not
        # expose TOGGLE actions until the mapping is confirmed.
        return {}

    # ---------------------------------------------------------
    # Score
    # ---------------------------------------------------------

    def team_score(self) -> float:
        if not hasattr(self.env, "calcGoalScore"):
            return 0.0

        scores = self.env.calcGoalScore()
        if scores is None or len(scores) <= TEAM_A_SCORE_INDEX:
            return 0.0
        return float(scores[TEAM_A_SCORE_INDEX])

    # ---------------------------------------------------------
    # Temporary V1 macro completion
    # ---------------------------------------------------------

    def apply_macro_action(
        self,
        agent_id: str,
        action: MacroAction,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply a completed high-level action to the full GameEnv state.

        V1 currently supports WAIT and COLLECT safely. SCORE and TOGGLE
        remain hidden by the action mask until their full GameEnv mappings
        are verified. This avoids inventing fake scoring rules that would
        later need to be rewritten.
        """
        if action.action_type == ActionType.WAIT:
            return {"success": True, "reason": "wait_complete"}

        if action.action_type == ActionType.COLLECT:
            return self._apply_collect(agent_id, payload)

        raise NotImplementedError(
            f"{action.action_type.value} execution is not connected to full GameEnv yet"
        )

    def _apply_collect(self, agent_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        channel = int(payload["channel"])
        row = int(payload["row"])
        col = int(payload["col"])

        board = self.env._board
        if board[channel, row, col] <= 0:
            return {"success": False, "reason": "object_already_taken"}

        agent = self.env._agents[agent_id]

        if np.asarray(agent.held_items).sum() > 0:
            return {"success": False, "reason": "agent_already_carrying"}

        board[channel, row, col] -= 1
        agent.held_items[channel] += 1
        agent.row = row
        agent.col = col

        return {
            "success": True,
            "reason": "collect_complete",
            "object_id": payload["object_id"],
            "channel": channel,
            "position": (row, col),
        }
