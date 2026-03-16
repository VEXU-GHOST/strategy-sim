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

    from push_back.env.render import get_render_timings, _timings

    _obs, _infos = env.reset(seed=seed, options={"balls": initial_balls})

    if out.suffix != ".mp4":
        out = out.with_suffix(".mp4")

    first_frame = env.render(draw_grid=grid)
    enc = _Encoder(first_frame.size, out)
    enc.feed(first_frame)
    frame_count = 1

    t0 = time.perf_counter()
    sim_dt: float = 0.0
    render_dt: float = 0.0
    encode_dt: float = 0.0
    for i in trange(steps, desc="simulating", miniters=steps // 20):
        t_sim = time.perf_counter()
        obs = BaseRobot.build_obs(env.state)
        actions: dict[str, int] = {
            name: robots[name].tick(obs, idx)
            for idx, name in enumerate(env.possible_agents)
        }
        _obs, _rewards, _terms, _truncs, _infos = env.step(actions)
        sim_dt += time.perf_counter() - t_sim
        t_render = time.perf_counter()
        frame = env.render(draw_grid=grid)
        render_dt += time.perf_counter() - t_render
        t_enc = time.perf_counter()
        enc.feed(frame)
        encode_dt += time.perf_counter() - t_enc
        frame_count += 1
    t_enc = time.perf_counter()
    enc.finish()
    drain_dt = time.perf_counter() - t_enc
    encode_dt += drain_dt
    total_dt = time.perf_counter() - t0
    typer.echo(
        f"sim: {sim_dt:.2f}s ({sim_dt / steps * 1000:.1f} ms/step) | "
        f"render: {render_dt:.2f}s ({render_dt / steps * 1000:.1f} ms/frame) | "
        f"encode: {encode_dt:.2f}s ({encode_dt / frame_count * 1000:.1f} ms/frame) | "
        f"drain: {drain_dt:.2f}s | "
        f"total: {total_dt:.2f}s"
    )
    rt = get_render_timings()
    n = rt["frames"] or 1
    typer.echo(
        f"  render breakdown (ms/frame): "
        f"bg_hash={rt['bg_hash']/n*1000:.1f} "
        f"bg_copy={rt['bg_copy']/n*1000:.1f} "
        f"balls={rt['balls']/n*1000:.1f} "
        f"agents={rt['agents']/n*1000:.1f} "
        f"hud_step={rt['hud_step']/n*1000:.1f} "
        f"hud_robot={rt['hud_robot']/n*1000:.1f} "
        f"hud_ball={rt['hud_ball']/n*1000:.1f} "
        f"tobytes={rt['tobytes']/n*1000:.1f}"
    )
    typer.echo(f"saved {frame_count} frames → {out}")


FPS = 10
_QUEUE_DEPTH = 30  # buffer up to N frames before blocking the main thread


class _Encoder:
    """Threaded ffmpeg encoder — writes happen on a background thread."""

    def __init__(self, size: tuple[int, int], out: Path) -> None:
        import fcntl
        import subprocess

        out.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
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
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        F_SETPIPE_SZ = 1031
        try:
            fcntl.fcntl(self._proc.stdin.fileno(), F_SETPIPE_SZ, 1024 * 1024)  # type: ignore[union-attr]
        except OSError:
            pass

        import queue
        import threading

        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _writer(self) -> None:
        """Drain queue and write to ffmpeg stdin (runs in background thread)."""
        pipe = self._proc.stdin
        while True:
            data = self._q.get()
            if data is None:
                break
            pipe.write(data)  # type: ignore[union-attr]

    def feed(self, frame: Image.Image) -> None:
        """Convert frame and enqueue for writing (may block if queue is full)."""
        import time

        import numpy as np

        from push_back.env.render import _timings

        _t = time.perf_counter()
        raw = bytes(memoryview(np.asarray(frame)))
        _timings["tobytes"] += time.perf_counter() - _t
        self._q.put(raw)

    def finish(self) -> None:
        """Signal writer thread to stop, wait for ffmpeg to finish."""
        self._q.put(None)
        self._thread.join()
        self._proc.stdin.close()  # type: ignore[union-attr]
        self._proc.wait()
        if self._proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {self._proc.returncode}")


if __name__ == "__main__":
    app()
