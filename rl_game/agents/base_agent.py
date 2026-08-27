"""
agents/base_agent.py — Abstract base class all agents must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Interface that every agent must satisfy.
    DQNAgent (and any future agent type) inherits from this.
    """

    def __init__(self, team: int, agent_id: str):
        self.team = team
        self.agent_id = agent_id

    @abstractmethod
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Choose an action given the current board state.

        Args:
            state  : board tensor (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            epsilon: exploration rate (0 = greedy, 1 = fully random)
        Returns:
            action int
        """

    @abstractmethod
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the agent's replay buffer."""

    @abstractmethod
    def train_step(self) -> float | None:
        """
        Sample from the replay buffer and perform one gradient update.
        Returns the training loss, or None if buffer isn't ready yet.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""
