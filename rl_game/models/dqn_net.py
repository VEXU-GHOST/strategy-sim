"""
models/dqn_net.py — Convolutional DQN network.

Input : (batch, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)  float tensor
Output: (batch, NUM_ACTIONS)  Q-values for each action
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOARD_SIZE, NUM_CHANNELS, NUM_ACTIONS


class DQNNet(nn.Module):
    """
    Small convolutional network suitable for CPU training.
    Scales well to GPU automatically via .to(device).

    Architecture:
        Conv block 1: NUM_CHANNELS → 32 filters, 8×8, stride 4
        Conv block 2: 32 → 64 filters, 4×4, stride 2
        Conv block 3: 64 → 64 filters, 3×3, stride 1
        Flatten → FC(512) → FC(NUM_ACTIONS)
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, num_actions: int = NUM_ACTIONS):
        super().__init__()

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        # )
        self.input_size = 345
        self.conv = nn.Sequential(
        nn.Linear(self.input_size, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_actions + 15)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape (batch, C, H, W), values in [0, 1]
        Returns:
            Q-values tensor of shape (batch, NUM_ACTIONS)
        """
        x = torch.flatten(x, start_dim=1)
        return self.conv(x)
