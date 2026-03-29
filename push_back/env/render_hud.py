"""Cached HUD panels and agent sprites for P-mode rendering.

Cache pyramid: render_*_panel → _render_text → _render_glyph
All cached arrays are (H, W) uint8 palette indices.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from push_back.env.render import (
    AGENT_RADIUS,
    FLAT_PALETTE,
    LINE_HEIGHT,
    _P_AGENTS,
    _P_BALL,
    _P_BALL_BLUE,
    _P_BALL_RED,
    _P_BLACK,
    _P_LABEL,
    _P_WHITE,
    _get_fonts,
)
from push_back.env.state import HEADING_DELTAS

# Sentinel palette index for transparent sprite background.
_P_TRANSPARENT: int = 255


# ── Glyph / text cache ──────────────────────────────────────────────────────


@lru_cache(maxsize=128)
def _render_glyph(char: str, color_idx: int) -> np.ndarray:
    """Single character as (LINE_HEIGHT, W) uint8 palette indices on black bg."""
    _, font_md, _ = _get_fonts()
    w = max(int(font_md.getlength(char)), 1)
    mask = Image.new("L", (w, LINE_HEIGHT), 0)
    ImageDraw.Draw(mask).text((0, 0), char, fill=255, font=font_md)
    arr = np.asarray(mask)
    result = np.zeros_like(arr, dtype=np.uint8)
    result[arr > 128] = color_idx
    return result


@lru_cache(maxsize=512)
def _render_text(text: str, color_idx: int) -> np.ndarray:
    """Text line as (LINE_HEIGHT, W) uint8 palette indices on black bg."""
    glyphs = [_render_glyph(c, color_idx) for c in text]
    w = sum(g.shape[1] for g in glyphs)
    result = np.zeros((LINE_HEIGHT, max(w, 1)), dtype=np.uint8)
    x = 0
    for g in glyphs:
        gw = g.shape[1]
        result[:, x : x + gw] = g
        x += gw
    return result


# ── HUD panel cache ──────────────────────────────────────────────────────────


@lru_cache(maxsize=128)
def render_step_panel(step: int) -> np.ndarray:
    """'Step N' panel as (H, W) uint8 indices with black bg."""
    text_arr = _render_text(f"Step {step}", _P_LABEL)
    h, w = text_arr.shape
    panel = np.zeros((h + 4, w + 4), dtype=np.uint8)
    panel[2 : 2 + h, 2 : 2 + w] = text_arr
    return panel


@lru_cache(maxsize=32)
def render_robot_panel(positions: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Robot position list as (H, W) uint8 indices with black bg."""
    n = len(positions)
    lines: list[np.ndarray] = []
    for i, (x, y) in enumerate(positions):
        ci = _P_AGENTS[i] if i < len(_P_AGENTS) else _P_LABEL
        lines.append(_render_text(f"R{i}: ({x}, {y})", ci))
    max_w = max(a.shape[1] for a in lines)
    panel = np.zeros((n * LINE_HEIGHT + 4, max_w + 4), dtype=np.uint8)
    for i, arr in enumerate(lines):
        panel[2 + i * LINE_HEIGHT : 2 + (i + 1) * LINE_HEIGHT, 2 : 2 + arr.shape[1]] = (
            arr
        )
    return panel


@lru_cache(maxsize=32)
def render_ball_panel(
    balls: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    """Ball position list as (H, W) uint8 indices with black bg."""
    if not balls:
        return np.zeros((1, 1), dtype=np.uint8)
    lines: list[np.ndarray] = []
    for idx, bx, by, bc in balls:
        ci = _P_BALL.get(bc, _P_LABEL)
        lines.append(_render_text(f"B{idx}: ({bx}, {by})", ci))
    max_w = max(a.shape[1] for a in lines)
    n = len(lines)
    panel = np.zeros((n * LINE_HEIGHT + 4, max_w + 4), dtype=np.uint8)
    for j, arr in enumerate(lines):
        panel[2 + j * LINE_HEIGHT : 2 + (j + 1) * LINE_HEIGHT, 2 : 2 + arr.shape[1]] = (
            arr
        )
    return panel


# ── Agent sprite cache ───────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def make_agent_sprite(
    heading: int,
    color_idx: int,
    red_n: int,
    blue_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cached P-mode agent sprite.

    Returns (sprite, mask):
        sprite: (H, W) uint8 palette indices
        mask:   (H, W) bool — True where opaque
    """
    r = int(AGENT_RADIUS)
    pad = 4
    size = 2 * r + 2 * pad
    img = Image.new("P", (size, size), _P_TRANSPARENT)
    img.putpalette(FLAT_PALETTE)
    draw = ImageDraw.Draw(img)
    cx, cy = r + pad, r + pad
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=color_idx,
        outline=_P_WHITE,
        width=2,
    )
    dx, dy = HEADING_DELTAS[heading]
    length = math.hypot(dx, dy) or 1.0
    lx = cx + int(r * (-dy) / length)
    ly = cy + int(r * (-dx) / length)
    draw.line([(cx, cy), (lx, ly)], fill=_P_WHITE, width=3)
    _, _, font_lg = _get_fonts()
    red_str, blue_str = str(red_n), str(blue_n)
    rw = font_lg.getlength(red_str)
    bw = font_lg.getlength(blue_str)
    ascent, descent = font_lg.getmetrics()
    total_w = rw + 4 + bw
    rx = cx - total_w / 2
    ty = cy - (ascent + descent) / 2
    draw.text(
        (rx, ty),
        red_str,
        fill=_P_BALL_RED,
        font=font_lg,
        stroke_width=2,
        stroke_fill=_P_BLACK,
    )
    draw.text(
        (rx + rw + 4, ty),
        blue_str,
        fill=_P_BALL_BLUE,
        font=font_lg,
        stroke_width=2,
        stroke_fill=_P_BLACK,
    )
    sprite_arr = np.array(img)
    mask_arr: np.ndarray = sprite_arr != _P_TRANSPARENT
    return sprite_arr, mask_arr
