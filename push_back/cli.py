"""CLI runner for push-back simulation.

Usage:
    push-back                       # 600 steps, output.mp4 at 10 fps
    push-back --steps 120           # custom step count
    push-back --out demo.mp4        # explicit path
    push-back --out demo            # .mp4 appended automatically
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="VEX field strategy simulator.", pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def run(
    steps: int = typer.Option(600, help="Simulation timesteps (0.1 s each)."),
    out: Path = typer.Option(
        Path("output/output"),
        help="Output path (.mp4 appended if no extension).",
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    grid: bool = typer.Option(False, "--grid", help="Draw grid lines on the field."),
    balls: Optional[str] = typer.Option(
        None,
        "--balls",
        help='Ball positions as "x,y,color;..." (color: 1=red, 2=blue). '
        "Overrides random placement.",
    ),
) -> None:
    """Run a simulation with default agents and save the render."""
    from push_back.env.robots import GoToRobot, RandomRobot, StandStill, SweeperRobot
    from push_back.env.push_back import PushBackEnv
    from push_back.env.robots.base import BaseRobot
    import numpy as np

    # Demo: red agents use macro GoTo actions, blue agents use primitives
    robots: dict[str, BaseRobot] = {
        # "red_0": GoToRobot(),
        "red_0": StandStill(),
        # "red_1": GoToRobot(),
        "red_1": StandStill(),
        "blue_0": SweeperRobot(),
        "blue_1": StandStill(),
    }
    env = PushBackEnv(robots=robots)

    initial_balls = None
    if balls is not None:
        rows = []
        for triple in balls.split(";"):
            parts = triple.strip().split(",")
            if len(parts) != 3:
                raise typer.BadParameter(f"Expected x,y,color triple, got {triple!r}")
            rows.append([int(parts[0]), int(parts[1]), int(parts[2])])
        initial_balls = np.array(rows, dtype=np.int32)

    from tqdm import trange
    import time

    _obs, _infos = env.reset(seed=seed, options={"balls": initial_balls})

    frames = [env.render(draw_grid=grid)]
    t0 = time.perf_counter()
    for i in trange(steps, desc="simulating", miniters=steps // 20):
        obs = BaseRobot.build_obs(env.state)
        actions: dict[str, int] = {
            name: robots[name].tick(obs, idx)
            for idx, name in enumerate(env.possible_agents)
        }
        _obs, _rewards, _terms, _truncs, _infos = env.step(actions)
        frame = env.render(draw_grid=grid)
        frames.append(frame)
    sim_dt = time.perf_counter() - t0
    typer.echo(f"sim: {sim_dt:.2f}s ({sim_dt / steps * 1000:.1f} ms/step)")

    if out.suffix != ".mp4":
        out = out.with_suffix(".mp4")
    t1 = time.perf_counter()
    _save(frames, out)
    save_dt = time.perf_counter() - t1
    typer.echo(f"saved {len(frames)} frames → {out} ({save_dt:.2f}s)")


FPS = 10


def _save(frames: list, out: Path) -> None:
    """Save frames as MP4 via ffmpeg (piped rawvideo)."""
    import subprocess

    import numpy as np

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


if __name__ == "__main__":
    app()
