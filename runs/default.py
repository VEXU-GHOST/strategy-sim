#!/usr/bin/env python3
"""Default simulation run — sweeper vs stand-still with random balls."""

from pathlib import Path

import numpy as np

from push_back.env.push_back import PushBackEnv
from push_back.env.robots import StandStill, SweeperRobot
from push_back.env.robots.base import BaseRobot
from push_back.runner import run_sim, save_frames

# ── Configuration ──────────────────────────────────────────────────────

STEPS: int = 600
SEED: int = 42
DRAW_GRID: bool = True
OUT: Path = Path("output/default.mp4")

# Robots — change these to try different strategies.
robots: dict[str, BaseRobot] = {
    "red_0": StandStill(),
    "red_1": StandStill(),
    "blue_0": SweeperRobot(),
    "blue_1": StandStill(),
}

# Ball placement — set to an (N, 3) int array of [x, y, color] rows to
# override random placement, or leave as None for random.
INITIAL_BALLS: np.ndarray | None = None
# Example:
# INITIAL_BALLS = np.array([[15, 15, 1], [20, 20, 2], [10, 25, 1]], dtype=np.int32)

# ── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = PushBackEnv(robots=robots)
    env.reset(seed=SEED, options={"balls": INITIAL_BALLS})

    frames = run_sim(env, robots, steps=STEPS, draw_grid=DRAW_GRID)
    save_frames(frames, OUT)
