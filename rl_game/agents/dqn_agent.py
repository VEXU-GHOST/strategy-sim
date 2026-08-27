"""
agents/dqn_agent.py — DQN agent with experience replay and a target network.

Each team has ONE shared DQNAgent instance; both agents on the team call
.act(), .remember(), and train together via .train_step().
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    NUM_ACTIONS,
    BATCH_SIZE,
    GAMMA,
    LR,
    REPLAY_BUFFER_SIZE,
)
from agents.base_agent import BaseAgent
from models.dqn_net import DQNNet
from training.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent.

    Parameters
    ----------
    team     : int   — TEAM_A or TEAM_B constant from config
    agent_id : str   — human-readable identifier, e.g. "TeamA"
    device   : str   — "cuda" | "cpu" | "auto"
    """

    def __init__(self, team: int, agent_id: str, device: str = "auto"):
        super().__init__(team, agent_id)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Online network (trained every TRAIN_EVERY steps)
        self.policy_net = DQNNet().to(self.device)

        # Target network (copied from policy_net periodically for stability)
        self.target_net = DQNNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — more stable than MSE

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def _state_to_numpy(self, state: dict | np.ndarray | torch.Tensor) -> np.ndarray:
        """Convert a raw environment state into a flat numeric feature vector."""
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()

        if isinstance(state, np.ndarray):
            return state.astype(np.float32).reshape(-1)

        if not isinstance(state, dict):
            return np.asarray(state, dtype=np.float32).reshape(-1)

        board_parts = []
        for board_chunk in state.get("board", []):
            for row in board_chunk:
                board_parts.append(np.asarray(row, dtype=np.float32).reshape(-1))

        board = np.concatenate(board_parts) if board_parts else np.array([], dtype=np.float32)
        # print(board.shape)
        # print("hi")
        goalst = np.asarray(state.get("goalStates", []), dtype=np.float32).reshape(-1)
        loadst = np.asarray(state.get("loaderStates", []), dtype=np.float32).reshape(-1)
        zonecl = np.asarray(state.get("zoneColors", []), dtype=np.float32).reshape(-1)

        agent_positions = state.get("agent_positions", {})
        if agent_positions:
            agpos = np.concatenate([
                np.asarray(pos, dtype=np.float32).reshape(-1)
                for pos in agent_positions.values()
            ])
        else:
            agpos = np.array([], dtype=np.float32)

        return np.concatenate([board, goalst, loadst, zonecl, agpos]).astype(np.float32)

    def act(self, state1: dict,  user_input: bool, epsilon: float = 0.0) -> int:
        """
        ε-greedy action selection.

        Args:
            state1 : environment state dict
            epsilon: probability of taking a random action
        Returns:
            action int
        """
        state = self._state_to_numpy(state1)

        if user_input:        
            while True:
                try:
                    action = int(input(f"Enter action for {self.agent_id}: "))
                    return action
                except ValueError:
                    print("Please enter a valid number")

        if random.random() < epsilon:
            return random.randrange(NUM_ACTIONS)

        self.policy_net.eval()
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            print(t.shape)
            q_values = self.policy_net(t)  # shape (1, NUM_ACTIONS)
        self.policy_net.train()
        action = int(q_values[0, :9].argmax().item())
        pick = int(q_values[0, 9:14].argmax().item())
        drop = int(q_values[0, 14:19].argmax().item())
        score = int(q_values[0, 19:25].argmax().item())
        action_num = 0
        action_num += score
        action_num*=6
        action_num += drop
        action_num*=6
        action_num += pick
        action_num*=9
        action_num = action
        
        return action_num

    def remember(
        self,
        state: dict | np.ndarray | torch.Tensor,
        action: int,
        reward: float,
        next_state: dict | np.ndarray | torch.Tensor,
        done: bool,
    ) -> None:
        state_features = self._state_to_numpy(state)
        print("errorrrrrrr")
        print(len(state_features))
        next_state_features = self._state_to_numpy(next_state)
        self.buffer.push(state_features, action, reward, next_state_features, done)

    def train_step(self) -> float | None:
        """
        Sample a random batch from the replay buffer and perform one
        gradient update on the policy network.

        Returns:
            float loss value, or None if buffer isn't ready.
        """
        if not self.buffer.is_ready:
            return None

        batch = self.buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        print(len(states[0]))
        states_t      = torch.from_numpy(np.stack(states)).float().to(self.device)
        next_states_t = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions_t     = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t       = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q-values for the actions actually taken
        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (Bellman equation)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + GAMMA * q_next * (1.0 - dones_t)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target_net(self) -> None:
        """Copy policy network weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"[{self.agent_id}] Checkpoint saved -> {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[{self.agent_id}] Checkpoint loaded <- {path}")
