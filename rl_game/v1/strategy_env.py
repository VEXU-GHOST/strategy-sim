from __future__ import annotations

from typing import Dict, Optional

from action_mask import get_available_actions
from config_v1 import ACTIVE_AGENTS, EPISODE_TIME
from executors.timed_sim_executor import TimedSimExecutor
from game_env_bridge import GameEnvBridge
from state_adapter import StateAdapter


class StrategyEnv:
    """
    High-level event-driven RL environment layered on top of the full GameEnv.

    The full simulator remains the source of truth for field state.
    V1 only reduces the observation/action interface seen by the RL policy.
    """

    def __init__(
        self,
        game_env,
        executor=None,
        state_adapter=None,
        episode_time: float = EPISODE_TIME,
    ):
        self.bridge = GameEnvBridge(game_env)
        self.executor = executor if executor is not None else TimedSimExecutor()
        self.state_adapter = state_adapter if state_adapter is not None else StateAdapter()
        self.episode_time = float(episode_time)
        self.time = 0.0

    def reset(self):
        self.time = 0.0
        self.executor.reset()
        self.bridge.reset()
        return self.get_state()

    def get_state(self):
        return self.state_adapter.build(
            bridge=self.bridge,
            executor=self.executor,
            current_time=self.time,
            episode_time=self.episode_time,
        )

    def get_available_actions(self, agent_id: str):
        if agent_id not in ACTIVE_AGENTS:
            raise ValueError(f"{agent_id} is not an active V1 learning agent")

        state = self.get_state()
        return get_available_actions(state, agent_id, self.bridge)

    def start_action(self, agent_id: str, action):
        available = self.get_available_actions(agent_id)
        if action not in available:
            raise ValueError(
                f"Action {action} is not available for {agent_id}. "
                f"Available: {[str(a) for a in available]}"
            )

        return self.executor.start_action(
            agent_id=agent_id,
            action=action,
            bridge=self.bridge,
            current_time=self.time,
        )

    def advance_to_next_event(self):
        score_before = self.bridge.team_score()

        next_time, completed = self.executor.advance_to_next_event(
            current_time=self.time,
            episode_time=self.episode_time,
        )
        self.time = float(next_time)

        completion_info: Dict[str, dict] = {}

        for agent_id, pending in completed.items():
            completion_info[agent_id] = self.bridge.apply_macro_action(
                agent_id=agent_id,
                action=pending.action,
                payload=pending.payload,
            )

        score_after = self.bridge.team_score()
        reward = float(score_after - score_before)

        done = self.done()
        info = {
            "time": self.time,
            "completed_agents": list(completed.keys()),
            "completion_info": completion_info,
        }

        return self.get_state(), reward, done, info

    def step(self, actions: Dict[str, object]):
        for agent_id, action in actions.items():
            self.start_action(agent_id, action)

        return self.advance_to_next_event()

    def done(self) -> bool:
        return self.time >= self.episode_time
