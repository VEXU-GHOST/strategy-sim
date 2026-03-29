"""CLI runner for push-back simulation.

Usage:
    push-back                       # 600 steps, output.mp4 at 10 fps
    push-back --steps 120           # custom step count
    push-back --out demo.mp4        # explicit path
    push-back --out demo            # .mp4 appended automatically
"""

from __future__ import annotations

from push_back.env.render_encoder import Encoder, FPS


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
    play: bool = typer.Option(
        False, "--play", help="Open output in mpv after rendering."
    ),
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
    from PIL import Image
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

    from push_back.env.perf import PerfAccum
    from push_back.env.render import get_render_perf

    _obs, _infos = env.reset(seed=seed, options={"balls": initial_balls})

    if out.suffix != ".mp4":
        out = out.with_suffix(".mp4")

    first_frame = env.render(draw_grid=grid)

    if play:
        import atexit
        import os
        import tempfile

        tmp_fd, _tmp_str = tempfile.mkstemp(suffix=".mp4", prefix="pb_preview_")
        os.close(tmp_fd)
        tmp_path = Path(_tmp_str)

        def _cleanup() -> None:
            if tmp_path.exists():
                typer.echo(f"cleaning up {tmp_path}")
                tmp_path.unlink()
            else:
                typer.echo(f"already cleaned up {tmp_path}")

        atexit.register(_cleanup)
        enc = Encoder(
            first_frame.size,
            tmp_path,
            codec="libx264",
            pix_fmt="yuv420p",
            preset="ultrafast",
        )
    else:
        enc = Encoder(first_frame.size, out)

    enc.feed(first_frame)
    frame_count = 1

    perf = PerfAccum()
    t0 = time.perf_counter()
    for i in trange(steps, desc="simulating", miniters=steps // 20):
        with perf.section("sim"):
            obs = BaseRobot.build_obs(env.state)
            actions: dict[str, int] = {
                name: robots[name].tick(obs, idx)
                for idx, name in enumerate(env.possible_agents)
            }
            _obs, _rewards, _terms, _truncs, _infos = env.step(actions)
        with perf.section("render"):
            frame = env.render(draw_grid=grid)
        with perf.section("encoder_feed"):
            enc.feed(frame)
        frame_count += 1
    with perf.section("drain"):
        enc.finish()
    enc.report()
    total_dt = time.perf_counter() - t0

    typer.echo(
        f"main (ms/frame): {perf.report_ms(frame_count)} "
        f"| total: {total_dt:.2f}s ({total_dt / frame_count * 1000:.2f} ms/frame)"
    )
    rp = get_render_perf()
    typer.echo(f"  render (ms/frame): {rp.report_ms(frame_count)}")
    typer.echo(f"saved {frame_count} frames → {tmp_path if play else out}")

    if play:
        from push_back.env.render_encoder import play_and_reencode

        play_and_reencode(tmp_path, out)
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app()
