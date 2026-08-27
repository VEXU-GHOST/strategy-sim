"""
training/replay_buffer.py — Fixed-size circular experience replay buffer.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np


Transition = Tuple[
    np.ndarray,  # state
    int,         # action
    float,       # reward
    np.ndarray,  # next_state
    bool,        # done
]


class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) transitions.
    Sampling is uniform random without replacement.
    """

    def __init__(self, capacity: int):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True once buffer has enough samples to fill at least one batch."""
        return len(self._buffer) >= 64  # uses BATCH_SIZE implicitly
