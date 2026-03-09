"""CLI runner for push-back simulation.

Usage:
    push-back                       # 600 steps (60 s), saves output.gif
    push-back --steps 120           # custom step count
    push-back --out frame.png       # saves frame_000.png, frame_001.png, …
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="VEX field strategy simulator.", pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def run(
    steps: int = typer.Option(600, help="Simulation timesteps (0.1 s each)."),
    out: Path = typer.Option(
        Path("output/output.png"), help="Output file path (.gif or .png)."
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    fps: int = typer.Option(10, help="Frames per second for GIF output."),
    grid: bool = typer.Option(False, "--grid", help="Draw grid lines on the field."),
) -> None:
    """Run a simulation with default agents and save the render."""
    from push_back.env.robots import GoToRobot, RandomRobot, StandStill
    from push_back.env.push_back import PushBackEnv
    from push_back.env.robots.base import BaseRobot

    # Demo: red agents use macro GoTo actions, blue agents use primitives
    robots: dict[str, BaseRobot] = {
        "red_0": GoToRobot(),
        "red_1": GoToRobot(),
        "blue_0": RandomRobot(),
        "blue_1": StandStill(),
    }
    env = PushBackEnv(robots=robots)
    _obs, _infos = env.reset(seed=seed)

    frames = [env.render(draw_grid=grid)]
    for i in range(steps):
        obs = BaseRobot.build_obs(env.state)
        actions: dict[str, int] = {
            name: robots[name].tick(obs, idx)
            for idx, name in enumerate(env.possible_agents)
        }
        _obs, _rewards, _terms, _truncs, _infos = env.step(actions)
        frame = env.render(draw_grid=grid)
        frames.append(frame)
        if (i + 1) % 10 == 0:
            typer.echo(f"step {i + 1}/{steps}")

    _save(frames, out, fps)
    typer.echo(f"saved {len(frames)} frames → {out}")


def _save(frames: list, out: Path, fps: int) -> None:
    """Save frames as GIF or numbered PNGs."""
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
        # save individual PNGs: out_000.png, out_001.png, …
        stem = out.with_suffix("")
        for i, img in enumerate(images):
            img.save(f"{stem}_{i:03d}.png")


if __name__ == "__main__":
    app()
