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
    enc.report()
    drain_dt = time.perf_counter() - t_enc
    encode_dt += drain_dt
    total_dt = time.perf_counter() - t0
    main_per_frame = (sim_dt + render_dt + encode_dt) / frame_count * 1000
    typer.echo(
        f"sim: {sim_dt:.2f}s ({sim_dt / steps * 1000:.2f} ms/step) | "
        f"render: {render_dt:.2f}s ({render_dt / steps * 1000:.2f} ms/frame) | "
        f"encode: {encode_dt:.2f}s ({encode_dt / frame_count * 1000:.2f} ms/frame) | "
        f"drain: {drain_dt:.2f}s | "
        f"total: {total_dt:.2f}s ({main_per_frame:.2f} ms/frame)"
    )
    rt = get_render_timings()
    n = rt["frames"] or 1
    typer.echo(
        f"  render breakdown (ms/frame): "
        f"bg_hash={rt['bg_hash']/n*1000:.2f} "
        f"bg_render={rt['bg_render']/n*1000:.2f} "
        f"bg_copy={rt['bg_copy']/n*1000:.2f} "
        f"balls={rt['balls']/n*1000:.2f} "
        f"agents={rt['agents']/n*1000:.2f} "
        f"hud_step_r={rt['hud_step_render']/n*1000:.2f} "
        f"hud_step_p={rt['hud_step_paste']/n*1000:.2f} "
        f"hud_robot_r={rt['hud_robot_render']/n*1000:.2f} "
        f"hud_robot_p={rt['hud_robot_paste']/n*1000:.2f} "
        f"hud_ball_r={rt['hud_ball_render']/n*1000:.2f} "
        f"hud_ball_p={rt['hud_ball_paste']/n*1000:.2f}"
    )
    typer.echo(
        f"  feed breakdown (ms/frame): "
        f"tobytes={rt['tobytes']/n*1000:.2f} "
        f"qput={rt.get('qput',0.0)/n*1000:.2f}"
    )
    typer.echo(f"saved {frame_count} frames → {out}")

    if play:
        import subprocess

        subprocess.run(["mpv", "--loop", str(out), "--pause", "--window-scale=2"])


FPS = 10
_QUEUE_DEPTH = 30  # buffer up to N frames before blocking the main thread


class _Encoder:
    """Threaded PyAV encoder — x264 runs in background thread, GIL released."""

    def __init__(self, size: tuple[int, int], out: Path) -> None:
        import queue
        import threading

        import av

        out.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self._container: av.container.OutputContainer = av.open(str(out), mode="w")
        self._stream: av.video.stream.VideoStream = self._container.add_stream(
            "libx264", rate=FPS
        )
        self._stream.width = w
        self._stream.height = h
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"preset": "veryfast"}

        self._q: queue.Queue[Image.Image | None] = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _writer(self) -> None:
        """Drain queue and encode via PyAV (GIL released during x264 work)."""
        import av
        import numpy as np
        import time

        stream = self._stream
        container = self._container
        # Per-phase accumulators (writer thread only — no lock needed)
        self._enc_timings: dict[str, float] = {
            "qget_wait": 0.0,
            "asarray": 0.0,
            "from_buf": 0.0,
            "encode": 0.0,
            "mux": 0.0,
            "flush": 0.0,
            "frames": 0,
        }
        et = self._enc_timings
        while True:
            t0 = time.perf_counter()
            pil_img = self._q.get()
            t1 = time.perf_counter()
            et["qget_wait"] += t1 - t0
            if pil_img is None:
                break
            npy = np.asarray(pil_img)  # zero-copy view into PIL buffer
            t2 = time.perf_counter()
            et["asarray"] += t2 - t1
            vf = av.VideoFrame.from_numpy_buffer(npy, format="rgb24")
            t3 = time.perf_counter()
            et["from_buf"] += t3 - t2
            for packet in stream.encode(vf):
                t4 = time.perf_counter()
                et["encode"] += t4 - t3
                container.mux(packet)
                t3 = time.perf_counter()
                et["mux"] += t3 - t4
            t4 = time.perf_counter()
            et["encode"] += t4 - t3  # encode call that yields no packet
            et["frames"] += 1
        # Flush encoder
        tf0 = time.perf_counter()
        for packet in stream.encode():
            container.mux(packet)
        et["flush"] = time.perf_counter() - tf0

    def feed(self, frame: Image.Image) -> None:
        """Enqueue PIL frame for encoding (may block if queue full)."""
        import time

        from push_back.env.render import _timings

        _t = time.perf_counter()
        # No tobytes needed — PyAV reads from PIL image directly
        _timings["tobytes"] += 0.0

        _t2 = time.perf_counter()
        self._q.put(frame)
        _timings["qput"] = _timings.get("qput", 0.0) + (time.perf_counter() - _t2)

    def finish(self) -> None:
        """Signal writer thread to stop, flush and close container."""
        self._q.put(None)
        self._thread.join()
        self._container.close()

    def report(self) -> None:
        """Print encoder thread timing breakdown."""
        et = self._enc_timings
        n = et["frames"] or 1
        import typer

        per_frame = (
            (
                et["qget_wait"]
                + et["asarray"]
                + et["from_buf"]
                + et["encode"]
                + et["mux"]
            )
            / n
            * 1000
        )
        typer.echo(
            f"  encoder thread (ms/frame): "
            f"qget={et['qget_wait']/n*1000:.2f} "
            f"asarray={et['asarray']/n*1000:.3f} "
            f"from_buf={et['from_buf']/n*1000:.3f} "
            f"encode={et['encode']/n*1000:.2f} "
            f"mux={et['mux']/n*1000:.3f} "
            f"| total={per_frame:.2f} "
            f"flush={et['flush']*1000:.1f}ms(total)"
        )


if __name__ == "__main__":
    app()
