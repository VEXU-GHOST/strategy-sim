"""Cached HUD overlay panels for the field renderer.

Each function returns an RGBA :class:`~PIL.Image.Image` that gets pasted onto
the frame.  Results are ``@lru_cache``-d so identical panel content across
consecutive frames is essentially free.

Cache pyramid: render_*_panel → _render_text → _render_glyph
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
def _render_glyph(char: str, color: tuple[int, int, int]) -> Image.Image:
    """RGBA sprite of one character, LINE_HEIGHT tall for easy composition."""
    w = max(int(_FONT_MD.getlength(char)), 1)
    img = Image.new("RGBA", (w, LINE_HEIGHT), (0, 0, 0, 255))
    ImageDraw.Draw(img).text((0, 0), char, fill=color, font=_FONT_MD)
    return img


@lru_cache(maxsize=512)
def _render_text(text: str, color: tuple[int, int, int]) -> Image.Image:
    """RGBA sprite of a text string, composed from cached per-char glyphs."""
    glyphs: list[Image.Image] = [_render_glyph(c, color) for c in text]
    w = sum(g.width for g in glyphs)
    img = Image.new("RGBA", (max(w, 1), LINE_HEIGHT), (0, 0, 0, 255))
    x = 0
    for g in glyphs:
        img.paste(g, (x, 0))
        x += g.width
    return img


@lru_cache(maxsize=128)
def render_step_panel(step: int) -> Image.Image:
    """Small RGBA image with 'Step N'."""
    text_img = _render_text(f"Step {step}", LABEL_COLOR)
    img = Image.new("RGBA", (text_img.width + 4, text_img.height + 4), (0, 0, 0, 255))
    img.paste(text_img, (2, 2))
    return img


@lru_cache(maxsize=32)
def render_robot_panel(
    positions: tuple[tuple[int, int], ...],
) -> Image.Image:
    """RGBA panel listing robot positions."""
    n = len(positions)
    lines: list[Image.Image] = []
    for i, (x, y) in enumerate(positions):
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (200, 200, 200)
        lines.append(_render_text(f"R{i}: ({x}, {y})", color))
    max_w = max(line.width for line in lines)
    img = Image.new("RGBA", (max_w + 4, n * LINE_HEIGHT + 4), (0, 0, 0, 255))
    for i, line in enumerate(lines):
        img.paste(line, (2, 2 + i * LINE_HEIGHT))
    return img


@lru_cache(maxsize=32)
def render_ball_panel(
    balls: tuple[tuple[int, int, int, int], ...],
) -> Image.Image:
    """RGBA panel listing ball positions (sorted red-first)."""
    if not balls:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    lines: list[Image.Image] = []
    for idx, bx, by, bc in balls:
        c = BALL_COLORS.get(bc, (200, 200, 200))
        lines.append(_render_text(f"B{idx}: ({bx}, {by})", c))
    max_w = max(line.width for line in lines)
    img = Image.new("RGBA", (max_w + 4, len(lines) * LINE_HEIGHT + 4), (0, 0, 0, 255))
    for j, line in enumerate(lines):
        img.paste(line, (2, 2 + j * LINE_HEIGHT))
    return img
