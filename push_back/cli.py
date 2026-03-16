"""CLI runner for push-back simulation.

Usage:
    push-back                       # 600 steps, saves output.gif + output_NNN.png
    push-back --steps 120           # custom step count
    push-back --out demo.gif        # GIF only
    push-back --out demo.png        # numbered PNGs only
    push-back --out demo            # both GIF and PNGs
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
        help="Output path. .gif → GIF, .png → numbered PNGs, no extension → both.",
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    fps: int = typer.Option(10, help="Frames per second for GIF output."),
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

    _obs, _infos = env.reset(seed=seed, options={"balls": initial_balls})

    frames = [env.render(draw_grid=grid)]
    for i in trange(steps, desc="simulating"):
        obs = BaseRobot.build_obs(env.state)
        actions: dict[str, int] = {
            name: robots[name].tick(obs, idx)
            for idx, name in enumerate(env.possible_agents)
        }
        _obs, _rewards, _terms, _truncs, _infos = env.step(actions)
        frame = env.render(draw_grid=grid)
        frames.append(frame)

    if out.suffix in (".gif", ".png"):
        _save(frames, out, fps)
        typer.echo(f"saved {len(frames)} frames → {out}")
    else:
        gif_path = out.with_suffix(".gif")
        png_path = out.with_suffix(".png")
        _save(frames, gif_path, fps)
        typer.echo(f"saved {len(frames)} frames → {gif_path}")
        _save(frames, png_path, fps)
        typer.echo(f"saved {len(frames)} frames → {png_path}")


def _save(frames: list, out: Path, fps: int) -> None:
    """Save frames as GIF or numbered PNGs."""
    from PIL import Image
    from tqdm import tqdm

    out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in tqdm(frames, desc="rendering")]

    if out.suffix == ".gif":
        images[0].save(
            out,
            save_all=True,
            append_images=tqdm(images[1:], desc="saving gif"),
            duration=1000 // fps,
            loop=0,
        )
    else:
        # save individual PNGs: out_000.png, out_001.png, …
        stem = out.with_suffix("")
        for i, img in tqdm(enumerate(images), total=len(images), desc="saving pngs"):
            img.save(f"{stem}_{i:03d}.png")


if __name__ == "__main__":
    app()
