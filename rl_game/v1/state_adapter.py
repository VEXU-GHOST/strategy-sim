from __future__ import annotations

from typing import Dict

from config_v1 import ACTIVE_AGENTS
from state_schema import StrategicState


class StateAdapter:
    """
    Convert the full GameEnv state into the smaller V1 strategic state.
    """

    def build(
        self,
        bridge,
        executor,
        current_time: float,
        episode_time: float,
    ) -> StrategicState:
        board_size = float(bridge.board_size)

        agents: Dict[str, dict] = {}

        for agent_id in ACTIVE_AGENTS:
            row, col = bridge.agent_position(agent_id)
            held_channels = bridge.held_item_channels(agent_id)

            action = executor.get_current_action(agent_id)

            agents[agent_id] = {
                "row": float(row),
                "col": float(col),
                "row_norm": float(row) / max(1.0, board_size - 1.0),
                "col_norm": float(col) / max(1.0, board_size - 1.0),
                "carrying": len(held_channels) > 0,
                "held_item_channels": held_channels,
                "busy": executor.is_busy(agent_id),
                "current_action": (
                    action.action_type.value if action is not None else None
                ),
                "target": action.target if action is not None else None,
                "remaining_time": executor.get_remaining_time(
                    agent_id,
                    current_time,
                ),
            }

        return {
            "time": float(current_time),
            "time_remaining": max(0.0, float(episode_time - current_time)),
            "team_score": float(bridge.team_score()),
            "agents": agents,
            "object_states": bridge.object_states(),
            "goal_states": bridge.goal_states(),
            "toggle_states": bridge.toggle_states(),
            "loader_states": bridge.loader_states(),
        }
