"""
config.py — Central configuration for the RL game framework.

Edit this file to tune hyperparameters, game settings, and training behaviour.
"""

from enum import IntEnum


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------
# Add new actions here freely — the rest of the framework reads NUM_ACTIONS
# automatically from the length of this enum.

class Action(IntEnum):
    UP     = 0
    DOWN   = 1
    LEFT   = 2
    RIGHT  = 3
    U_L = 4
    U_R = 5
    D_L = 6
    D_R = 7
    STAY = 8
    # ← Add more actions here, e.g.:
    # ATTACK = 6
    # DEFEND = 7

NUM_ACTIONS: int = 25


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
BOARD_SIZE: int = 140          # Board is BOARD_SIZE × BOARD_SIZE
NUM_CHANNELS: int = 5         # Channels in board tensor:
                              #   0 = empty, 1 = team-A agents,
                              #   2 = team-B agents, 3+ = custom objects

AGENT_RADIUS: int = 2              # Agent radius in grid cells
NUM_TEAMS: int = 2
AGENTS_PER_TEAM: int = 2
NUM_AGENTS: int = NUM_TEAMS * AGENTS_PER_TEAM   # 4 total

# Team identifiers
TEAM_A: int = 0
TEAM_B: int = 1

# Starting positions for each agent [row, col]
# Override these or implement random spawn logic in GameEnv.reset()
SPAWN_POSITIONS: dict = {
    "A0": (23, 29),
    "A1": (29, 23),
    "B0": (3, 15),
    "B1": (15, 3),
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPISODES: int = 500           # Total training episodes
STEPS_PER_EPISODE: int = 200  # Max steps before episode ends
TRAIN_EVERY: int = 10         # Steps between gradient updates
BATCH_SIZE: int = 64          # Samples per training batch

GAMMA: float = 0.99           # Discount factor
LR: float = 1e-3              # Adam learning rate

EPSILON_START: float = 1.0    # Starting exploration rate
EPSILON_END: float = 0.05     # Minimum exploration rate
EPSILON_DECAY: float = 0.995  # Multiplicative decay per episode

REPLAY_BUFFER_SIZE: int = 10_000   # Max transitions stored
TARGET_UPDATE_EVERY: int = 20      # Episodes between target-net syncs


# ---------------------------------------------------------------------------
# Checkpointing & Logging
# ---------------------------------------------------------------------------
CHECKPOINT_DIR: str = "checkpoints"
CHECKPOINT_EVERY: int = 50    # Save model every N episodes
LOG_EVERY: int = 10           # Print stats every N episodes
