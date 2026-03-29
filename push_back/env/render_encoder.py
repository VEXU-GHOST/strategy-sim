"""Threaded PyAV video encoder + mpv preview with background re-encode."""

from __future__ import annotations
from time import perf_counter

MODULE_LOADED_TIME = perf_counter()


from pathlib import Path

from PIL import Image

FPS = 10
_QUEUE_DEPTH = 30  # buffer up to N frames before blocking the main thread


def play_and_reencode(tmp_path: Path, out: Path, fps: int = FPS) -> None:
    """Play tmp_path in mpv immediately; background-reencode to out, then hot-swap."""
    import os
    import threading

    import mpv
    import typer

    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"mpv ready at {perf_counter() - MODULE_LOADED_TIME:.3f}s")
    player = mpv.MPV(
        loop="inf",
        window_scale=2,
        pause=True,
        input_default_bindings=True,
        input_vo_keyboard=True,
        osc=True,
    )
    player.play(str(tmp_path))

    def _reencode() -> None:
        import av

        out.unlink(missing_ok=True)
        inp = av.open(str(tmp_path))
        outp = av.open(str(out), mode="w")
        in_stream = inp.streams.video[0]
        out_stream = outp.add_stream("libx264", rate=fps)
        out_stream.width = in_stream.width
        out_stream.height = in_stream.height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"preset": "veryslow", "crf": "28", "tune": "animation"}

        for frame in inp.decode(video=0):
            for pkt in out_stream.encode(frame):
                outp.mux(pkt)
        for pkt in out_stream.encode():
            outp.mux(pkt)
        outp.close()
        inp.close()

        old_sz = os.path.getsize(tmp_path)
        new_sz = os.path.getsize(out)
        # Hot-swap: save position, load new file at same pos with same pause state
        time_pos: float = player.time_pos
        if time_pos is None:
            time_pos = 0.0
            typer.echo("warning: time_pos is None at hot-swap — may have lost position")
        was_paused: bool = player.pause  # type: ignore[assignment]
        opts = f"start={time_pos:.3f},pause={'yes' if was_paused else 'no'}"
        typer.echo(f"hot-swap: loadfile {out.resolve()} replace {opts}")
        player.command("loadfile", str(out.resolve()), "replace", opts)
        import time

        while player.duration is None or player.duration <= 0:
            time.sleep(0.25)
        os.unlink(tmp_path)
        typer.echo(
            f"re-encoded: {old_sz:,} → {new_sz:,} bytes ({new_sz/old_sz:.0%}). Deleted {tmp_path}."
        )

    threading.Thread(target=_reencode, daemon=True).start()
    player.wait_for_shutdown()


class Encoder:
    """Threaded PyAV encoder — x264 runs in background thread, GIL released."""

    def __init__(
        self,
        size: tuple[int, int],
        out: Path,
        codec: str = "libx264",
        pix_fmt: str = "yuv420p",
        preset: str = "fast",
    ) -> None:
        import queue
        import threading

        import av

        out.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self._container: av.container.OutputContainer = av.open(str(out), mode="w")
        self._stream: av.video.stream.VideoStream = self._container.add_stream(
            codec, rate=FPS
        )
        self._stream.width = w
        self._stream.height = h
        self._stream.pix_fmt = pix_fmt
        self._stream.options = {"preset": preset, "tune": "animation"}

        self._q: queue.Queue[Image.Image | None] = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _writer(self) -> None:
        """Drain queue and encode via PyAV (GIL released during x264 work)."""
        import av
        import numpy as np

        from push_back.env.perf import PerfAccum

        stream = self._stream
        container = self._container
        self._perf = PerfAccum()
        self._frame_count: int = 0
        perf = self._perf
        pal_bgra: bytes | None = None

        while True:
            with perf.section("qget"):
                pil_img = self._q.get()
            if pil_img is None:
                break
            with perf.section("asarray"):
                indices = np.asarray(pil_img)  # (H, W) uint8 palette indices
            with perf.section("from_buf"):
                h, w = indices.shape
                vf = av.VideoFrame(w, h, "pal8")
                vf.planes[0].update(indices.tobytes())
                if pal_bgra is None:
                    palette_flat = pil_img.getpalette()
                    assert palette_flat is not None
                    pal_arr = np.zeros((256, 4), dtype=np.uint8)
                    pal_rgb = np.array(palette_flat, dtype=np.uint8).reshape(-1, 3)
                    pal_arr[: len(pal_rgb), 0] = pal_rgb[:, 2]  # B
                    pal_arr[: len(pal_rgb), 1] = pal_rgb[:, 1]  # G
                    pal_arr[: len(pal_rgb), 2] = pal_rgb[:, 0]  # R
                    pal_arr[: len(pal_rgb), 3] = 255  # A
                    pal_bgra = pal_arr.tobytes()
                vf.planes[1].update(pal_bgra)
            with perf.section("encode"):
                packets = list(stream.encode(vf))
            with perf.section("mux"):
                for packet in packets:
                    container.mux(packet)
            self._frame_count += 1

        with perf.section("flush"):
            for packet in stream.encode():
                container.mux(packet)

    def feed(self, frame: Image.Image) -> None:
        """Enqueue PIL frame for encoding (may block if queue full)."""
        self._q.put(frame)

    def finish(self) -> None:
        """Signal writer thread to stop, flush and close container."""
        self._q.put(None)
        self._thread.join()
        self._container.close()

    def report(self) -> None:
        """Print encoder thread timing breakdown."""
        import typer

        n = self._frame_count or 1
        perf = self._perf
        flush_ms = perf.seconds("flush") * 1000
        per_frame = (perf.total() - perf.seconds("flush")) / n * 1000
        typer.echo(
            f"  encoder ({n} frames, ms/frame): "
            f"{perf.report_ms(n, exclude=frozenset({'flush'}))} "
            f"| total={per_frame:.2f} flush={flush_ms:.1f}ms"
        )
