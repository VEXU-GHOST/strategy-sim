"""Cached HUD overlay panels for the field renderer.

Each function returns an RGBA :class:`~PIL.Image.Image` that gets pasted onto
the frame.  Results are ``@lru_cache``-d so identical panel content across
consecutive frames is essentially free.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from push_back.env.render import PPI
from push_back.env.state import BallColor

BALL_COLORS: dict[int, tuple[int, int, int]] = {
    BallColor.RED: (220, 40, 40),
    BallColor.BLUE: (40, 80, 220),
}
AGENT_COLORS: tuple[tuple[int, int, int], ...] = (
    (220, 50, 50),
    (255, 120, 120),
    (50, 80, 220),
    (120, 150, 255),
)
LABEL_COLOR: tuple[int, int, int] = (200, 200, 200)
LINE_HEIGHT: int = max(14, int(22 * PPI / 5))
_FONT_MD: ImageFont.FreeTypeFont = ImageFont.load_default(
    size=max(10, int(20 * PPI / 5))
)


@lru_cache(maxsize=128)
def render_step_panel(step: int) -> Image.Image:
    """Small RGBA image with 'Step N' on a black background."""
    text = f"Step {step}"
    bbox = _FONT_MD.getbbox(text)
    w, h = bbox[2] + 4, bbox[3] + 4
    img = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), text, fill=LABEL_COLOR, font=_FONT_MD)
    return img


@lru_cache(maxsize=32)
def render_robot_panel(
    positions: tuple[tuple[int, int], ...],
) -> Image.Image:
    """RGBA panel listing robot positions."""
    n = len(positions)
    labels: list[str] = [f"R{i}: ({x}, {y})" for i, (x, y) in enumerate(positions)]
    max_w = max(_FONT_MD.getbbox(l)[2] for l in labels)
    w = max_w + 4
    h = n * LINE_HEIGHT + 4
    img = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    for i, label in enumerate(labels):
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (200, 200, 200)
        draw.text((2, 2 + i * LINE_HEIGHT), label, fill=color, font=_FONT_MD)
    return img


@lru_cache(maxsize=32)
def render_ball_panel(
    balls: tuple[tuple[int, int, int, int], ...],
) -> Image.Image:
    """RGBA panel listing ball positions (sorted red-first)."""
    if not balls:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    labels: list[tuple[str, tuple[int, int, int]]] = []
    for idx, bx, by, bc in balls:
        c = BALL_COLORS.get(bc, (200, 200, 200))
        labels.append((f"B{idx}: ({bx}, {by})", c))
    max_w = max(_FONT_MD.getbbox(lbl)[2] for lbl, _ in labels)
    w = max_w + 4
    h = len(labels) * LINE_HEIGHT + 4
    img = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    for j, (lbl, c) in enumerate(labels):
        draw.text((2, 2 + j * LINE_HEIGHT), lbl, fill=c, font=_FONT_MD)
    return img
