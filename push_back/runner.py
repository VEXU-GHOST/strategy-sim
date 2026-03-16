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


FPS = 10


def save_frames(frames: list[np.ndarray], out: Path) -> None:
    """Save frames as MP4 via ffmpeg (piped rawvideo)."""
    import subprocess

    if out.suffix != ".mp4":
        out = out.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(FPS),
        "-i",
        "pipe:",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.asarray(frame).tobytes())  # type: ignore[union-attr]
    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    print(f"saved {len(frames)} frames → {out}")
