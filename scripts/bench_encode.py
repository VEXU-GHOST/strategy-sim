"""Benchmark render→encode pipeline variations.

Tests combinations of:
  - PIL image mode (RGB, RGBA)
  - Row stride alignment (none, 32-byte)
  - PyAV codec / pixel format (libx264/yuv420p, libx264rgb/rgb24, libx264/yuv444p)
  - Frame ingestion method (from_numpy_buffer, from_ndarray)

Runs each config for FRAMES frames × REPS repetitions, reports ms/frame for
the encode-thread portion (asarray + from_buf + encode + mux).
"""

from __future__ import annotations

import gc
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants — tune for reliability
# ---------------------------------------------------------------------------
FRAMES = 1200  # frames per trial (≈2 min of 10fps video)
REPS = 3  # repetitions per config
FPS = 10
W, H = 576, 480  # matches actual render output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_test_frames(n: int, mode: str, align: int) -> list[Image.Image]:
    """Pre-render n distinct PIL frames so rendering cost is excluded."""
    rng = np.random.default_rng(42)
    frames: list[Image.Image] = []
    channels = 4 if mode == "RGBA" else 3
    for i in range(n):
        if mode == "P":
            # Build an RGB frame, then quantize to palette
            arr = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            rgb_img = Image.fromarray(arr, mode="RGB")
            frames.append(
                rgb_img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
            )
            continue
        if align > 1:
            row_bytes = W * channels
            stride = ((row_bytes + align - 1) // align) * align
            buf = np.zeros((H, stride), dtype=np.uint8)
            buf[:, :row_bytes] = rng.integers(0, 255, (H, row_bytes), dtype=np.uint8)
            arr = np.lib.stride_tricks.as_strided(
                buf, shape=(H, W, channels), strides=(stride, channels, 1)
            )
            arr = np.ascontiguousarray(arr)
        else:
            arr = rng.integers(0, 255, (H, W, channels), dtype=np.uint8)
        frames.append(Image.fromarray(arr, mode=mode))
    return frames


@dataclass(frozen=True)
class Config:
    label: str
    pil_mode: str  # "RGB" or "RGBA"
    align: int  # 1 = no special alignment, 32 = 32-byte
    codec: str  # "libx264" or "libx264rgb"
    pix_fmt: str  # "yuv420p", "yuv444p", "rgb24"
    preset: str  # "veryfast" or "ultrafast"
    ingest: str  # "from_numpy_buffer", "from_ndarray", or "pal8"


def _av_fmt_for_mode(mode: str) -> str:
    return "rgba" if mode == "RGBA" else "rgb24"


def _run_encode(
    frames: list[Image.Image],
    cfg: Config,
    out_path: Path,
) -> dict[str, float]:
    """Encode frames with given config, return timing dict."""
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream(cfg.codec, rate=FPS)
    stream.width = W
    stream.height = H
    stream.pix_fmt = cfg.pix_fmt
    stream.options = {"preset": cfg.preset}

    timings: dict[str, float] = {
        "asarray": 0.0,
        "convert": 0.0,
        "ingest": 0.0,
        "encode": 0.0,
        "mux": 0.0,
    }

    for pil_img in frames:
        t0 = time.perf_counter()
        t1 = t0

        if cfg.ingest == "pal8":
            # Let ffmpeg/swscale do the pal8 → yuv conversion
            indices = np.asarray(pil_img)  # (H, W) uint8
            palette_flat = pil_img.getpalette()  # [r,g,b,...] 768 entries
            assert palette_flat is not None
            t1 = time.perf_counter()
            timings["convert"] += t1 - t0

            vf = av.VideoFrame(W, H, "pal8")
            # Copy index data into the frame's plane 0
            vf.planes[0].update(indices.tobytes())
            # Build BGRA palette (ffmpeg pal8 expects 256×4 BGRA)
            pal_arr = np.zeros((256, 4), dtype=np.uint8)
            pal_rgb = np.array(palette_flat, dtype=np.uint8).reshape(-1, 3)
            pal_arr[: len(pal_rgb), 0] = pal_rgb[:, 2]  # B
            pal_arr[: len(pal_rgb), 1] = pal_rgb[:, 1]  # G
            pal_arr[: len(pal_rgb), 2] = pal_rgb[:, 0]  # R
            pal_arr[: len(pal_rgb), 3] = 255  # A
            vf.planes[1].update(pal_arr.tobytes())
            t3 = time.perf_counter()
            timings["ingest"] += t3 - t1
        else:
            # Convert to RGB if needed (P or RGBA → RGB)
            needs_convert = cfg.pil_mode == "P" or (
                cfg.pil_mode == "RGBA" and cfg.pix_fmt != "rgba"
            )
            if needs_convert:
                rgb = pil_img.convert("RGB")
                npy = np.asarray(rgb)
                av_fmt = "rgb24"
            else:
                npy = np.asarray(pil_img)
                av_fmt = _av_fmt_for_mode(cfg.pil_mode)
            t1 = time.perf_counter()
            timings["convert"] += t1 - t0

            if cfg.ingest == "from_numpy_buffer":
                vf = av.VideoFrame.from_numpy_buffer(npy, format=av_fmt)
            else:
                vf = av.VideoFrame.from_ndarray(npy, format=av_fmt)
            t3 = time.perf_counter()
            timings["ingest"] += t3 - t1

        packets = list(stream.encode(vf))
        t4 = time.perf_counter()
        timings["encode"] += t4 - t3

        for pkt in packets:
            container.mux(pkt)
        timings["mux"] += time.perf_counter() - t4

    # flush
    t0 = time.perf_counter()
    for pkt in stream.encode():
        container.mux(pkt)
    timings["flush"] = time.perf_counter() - t0

    container.close()
    timings["file_kb"] = os.path.getsize(out_path) / 1024
    return timings


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
CONFIGS: list[Config] = [
    # --- baseline: current production path ---
    Config(
        "RGB/yuv420p/vfast/buf",
        "RGB",
        1,
        "libx264",
        "yuv420p",
        "veryfast",
        "from_numpy_buffer",
    ),
    Config(
        "RGB/yuv420p/vfast/ndarr",
        "RGB",
        1,
        "libx264",
        "yuv420p",
        "veryfast",
        "from_ndarray",
    ),
    # --- alignment ---
    Config(
        "RGB/yuv420p/vfast/buf/a32",
        "RGB",
        32,
        "libx264",
        "yuv420p",
        "veryfast",
        "from_numpy_buffer",
    ),
    # --- ultrafast preview path ---
    Config(
        "RGB/rgb24/ufast/buf",
        "RGB",
        1,
        "libx264rgb",
        "rgb24",
        "ultrafast",
        "from_numpy_buffer",
    ),
    Config(
        "RGB/rgb24/ufast/ndarr",
        "RGB",
        1,
        "libx264rgb",
        "rgb24",
        "ultrafast",
        "from_ndarray",
    ),
    Config(
        "RGB/rgb24/ufast/buf/a32",
        "RGB",
        32,
        "libx264rgb",
        "rgb24",
        "ultrafast",
        "from_numpy_buffer",
    ),
    # --- yuv444p (lossless chroma) ---
    Config(
        "RGB/yuv444p/vfast/buf",
        "RGB",
        1,
        "libx264",
        "yuv444p",
        "veryfast",
        "from_numpy_buffer",
    ),
    # --- RGBA source ---
    Config(
        "RGBA/yuv420p/vfast/buf",
        "RGBA",
        1,
        "libx264",
        "yuv420p",
        "veryfast",
        "from_numpy_buffer",
    ),
    Config(
        "RGBA/rgb24/ufast/buf",
        "RGBA",
        1,
        "libx264rgb",
        "rgb24",
        "ultrafast",
        "from_numpy_buffer",
    ),
    # --- ultrafast yuv420p (fast encode, small file?) ---
    Config(
        "RGB/yuv420p/ufast/buf",
        "RGB",
        1,
        "libx264",
        "yuv420p",
        "ultrafast",
        "from_numpy_buffer",
    ),
    # --- veryfast rgb24 ---
    Config(
        "RGB/rgb24/vfast/buf",
        "RGB",
        1,
        "libx264rgb",
        "rgb24",
        "veryfast",
        "from_numpy_buffer",
    ),
    # --- P (palette) source ---
    Config(
        "P/yuv420p/ufast/buf",
        "P",
        1,
        "libx264",
        "yuv420p",
        "ultrafast",
        "from_numpy_buffer",
    ),
    Config(
        "P/yuv420p/vfast/buf",
        "P",
        1,
        "libx264",
        "yuv420p",
        "veryfast",
        "from_numpy_buffer",
    ),
    Config(
        "P/rgb24/ufast/buf",
        "P",
        1,
        "libx264rgb",
        "rgb24",
        "ultrafast",
        "from_numpy_buffer",
    ),
    # --- P with ffmpeg pal8→yuv conversion (no PIL .convert) ---
    Config("P/yuv420p/ufast/pal8", "P", 1, "libx264", "yuv420p", "ultrafast", "pal8"),
    Config("P/yuv420p/vfast/pal8", "P", 1, "libx264", "yuv420p", "veryfast", "pal8"),
]


def main() -> None:
    print(f"Benchmarking {len(CONFIGS)} configs × {REPS} reps × {FRAMES} frames")
    print(f"Frame size: {W}×{H}")
    print()

    # Pre-generate frames for both modes
    print("Generating test frames (RGB)...", flush=True)
    rgb_frames_1 = _make_test_frames(FRAMES, "RGB", 1)
    print("Generating test frames (RGB, 32-byte aligned)...", flush=True)
    rgb_frames_32 = _make_test_frames(FRAMES, "RGB", 32)
    print("Generating test frames (RGBA)...", flush=True)
    rgba_frames_1 = _make_test_frames(FRAMES, "RGBA", 1)
    print("Generating test frames (P palette)...", flush=True)
    p_frames_1 = _make_test_frames(FRAMES, "P", 1)
    print()

    frame_sets: dict[tuple[str, int], list[Image.Image]] = {
        ("RGB", 1): rgb_frames_1,
        ("RGB", 32): rgb_frames_32,
        ("RGBA", 1): rgba_frames_1,
        ("P", 1): p_frames_1,
    }

    results: dict[str, list[dict[str, float]]] = {}

    for cfg in CONFIGS:
        print(f"  {cfg.label}", end="", flush=True)
        frames = frame_sets[(cfg.pil_mode, cfg.align)]
        trials: list[dict[str, float]] = []
        for rep in range(REPS):
            gc.collect()
            gc.disable()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp = Path(f.name)
            try:
                t = _run_encode(frames, cfg, tmp)
                trials.append(t)
                print(".", end="", flush=True)
            finally:
                tmp.unlink(missing_ok=True)
                gc.enable()
        results[cfg.label] = trials
        print()

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 120)
    hdr = (
        f"{'Config':<35s} "
        f"{'convert':>8s} {'ingest':>8s} "
        f"{'encode':>8s} {'mux':>8s} {'total':>8s} "
        f"{'file_KB':>8s}"
    )
    print(hdr)
    print("-" * 120)

    for label, trials in results.items():
        # Average across reps
        keys = ["convert", "ingest", "encode", "mux"]
        avgs: dict[str, float] = {}
        for k in keys:
            avgs[k] = sum(t[k] for t in trials) / len(trials) / FRAMES * 1000
        total = sum(avgs.values())
        file_kb = sum(t["file_kb"] for t in trials) / len(trials)
        print(
            f"{label:<35s} "
            f"{avgs['convert']:8.3f} {avgs['ingest']:8.3f} "
            f"{avgs['encode']:8.3f} {avgs['mux']:8.3f} {total:8.3f} "
            f"{file_kb:8.1f}"
        )

    print("-" * 120)
    print("All times in ms/frame (lower is better). file_KB = average output size.")


if __name__ == "__main__":
    main()
