from abc import ABC, abstractmethod


class ActionExecutor(ABC):
    """
    Interface for executing high-level strategy actions.

    V1 uses TimedSimExecutor. Later this can be replaced by a
    low-level GameEnv skill executor or a real-robot executor while
    keeping the RL policy and StrategyEnv interface unchanged.
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def start_action(self, agent_id, action, bridge, current_time: float):
        pass

    @abstractmethod
    def advance_to_next_event(self, current_time: float, episode_time: float):
        pass

    @abstractmethod
    def is_busy(self, agent_id: str) -> bool:
        pass

    @abstractmethod
    def get_remaining_time(self, agent_id: str, current_time: float) -> float:
        pass

    @abstractmethod
    def get_current_action(self, agent_id: str):
        pass
