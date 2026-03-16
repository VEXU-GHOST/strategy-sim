"""Simulation runner utilities shared by scripts in ``runs/``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from push_back.env.push_back import PushBackEnv
from push_back.env.robots.base import BaseRobot


def run_sim(
    env: PushBackEnv,
    robots: dict[str, BaseRobot],
    *,
    steps: int = 600,
    draw_grid: bool = False,
) -> list[np.ndarray]:
    """Step the environment and collect rendered frames.

    Returns a list of HWC uint8 numpy arrays (one per timestep + initial).
    """
    frames: list[np.ndarray] = [env.render(draw_grid=draw_grid)]
    for i in range(steps):
        obs = BaseRobot.build_obs(env.state)
        actions: dict[str, int] = {
            name: robots[name].tick(obs, idx)
            for idx, name in enumerate(env.possible_agents)
        }
        env.step(actions)
        frames.append(env.render(draw_grid=draw_grid))
        if (i + 1) % 10 == 0:
            print(f"step {i + 1}/{steps}")
    return frames


def save_frames(frames: list[np.ndarray], out: Path, fps: int = 10) -> None:
    """Save frames as an animated GIF or numbered PNGs."""
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]

    if out.suffix == ".gif":
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=1000 // fps,
            loop=0,
        )
    else:
        stem = out.with_suffix("")
        for i, img in enumerate(images):
            img.save(f"{stem}_{i:03d}.png")

    print(f"saved {len(frames)} frames → {out}")
