"""Pillow-based renderer for WorldState.

Draws a top-down view of the 144×144" VEX field:
  - gray field with white border
  - colored circles for balls
  - direction indicators for each robot
"""

from __future__ import annotations

import threading
from collections import defaultdict
from functools import lru_cache

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from push_back.env.state import (
    BALL_DIA,
    BallColor,
    CELL_SIZE,
    FIELD_INCHES,
    GRID_SIZE,
    ROBOT_RADIUS,
    WorldState,
)

# pixels per inch — controls output resolution
PPI = 3

# colors
FIELD_COLOR = (80, 80, 80)
BORDER_COLOR = (220, 220, 220)
BALL_COLORS: dict[int, tuple[int, int, int]] = {
    BallColor.RED: (220, 40, 40),
    BallColor.BLUE: (40, 80, 220),
}
AGENT_COLORS: list[tuple[int, int, int]] = [
    (220, 50, 50),  # red 0
    (255, 120, 120),  # red 1
    (50, 80, 220),  # blue 0
    (120, 150, 255),  # blue 1
]
AGENT_RADIUS = ROBOT_RADIUS * CELL_SIZE * PPI  # robot radius in pixels
GOAL_TUBE_WIDTH = 3  # line width for goal tube outline
LABEL_COLOR = (200, 200, 200)
# per-side margins (pixels) — only as wide as the content on that edge
MARGIN_LEFT = 24 * PPI  # room for HUD sidebar
MARGIN_TOP = 8 * PPI
MARGIN_RIGHT = 20 * PPI  # room for origin (0,0) label
MARGIN_BOTTOM = 8 * PPI
GRID_COLOR = (100, 100, 100)
COLLISION_SEG_COLOR = (200, 200, 50)

# ── Fixed 256-color palette for P-mode rendering ──
# All colors used in the renderer must have an entry here.
_PALETTE_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),  # 0 — black
    FIELD_COLOR,  # 1
    BORDER_COLOR,  # 2
    GRID_COLOR,  # 3
    BALL_COLORS[BallColor.RED],  # 4
    BALL_COLORS[BallColor.BLUE],  # 5
    AGENT_COLORS[0],  # 6
    AGENT_COLORS[1],  # 7
    AGENT_COLORS[2],  # 8
    AGENT_COLORS[3],  # 9
    (255, 255, 255),  # 10 — white
    LABEL_COLOR,  # 11
    COLLISION_SEG_COLOR,  # 12
    (180, 180, 180),  # 13 — goal tube line
    (120, 120, 120),  # 14 — goal slot outline
)
_P_BLACK, _P_FIELD, _P_BORDER, _P_GRID = 0, 1, 2, 3
_P_BALL_RED, _P_BALL_BLUE = 4, 5
_P_AGENTS: tuple[int, ...] = (6, 7, 8, 9)
_P_WHITE, _P_LABEL, _P_COLLISION = 10, 11, 12
_P_GOAL, _P_GOAL_SLOT = 13, 14
_P_BALL: dict[int, int] = {BallColor.RED: _P_BALL_RED, BallColor.BLUE: _P_BALL_BLUE}

# Flat palette for PIL putpalette() — [r0,g0,b0, r1,g1,b1, ...]
FLAT_PALETTE: list[int] = []
for _c in _PALETTE_COLORS:
    FLAT_PALETTE.extend(_c)
FLAT_PALETTE.extend([0] * (768 - len(FLAT_PALETTE)))

LINE_HEIGHT: int = max(14, int(22 * PPI / 5))

# Font sizes — scaled to PPI (sizes tuned at PPI=5, scale linearly)
_FONT_SM_SIZE: int = max(8, int(12 * PPI / 5))
_FONT_MD_SIZE: int = max(10, int(20 * PPI / 5))
_FONT_LG_SIZE: int = max(12, int(34 * PPI / 5))

# FreeType font objects are NOT thread-safe.  Each thread gets its own set
# via threading.local so render workers never contend.
_tls = threading.local()


def _get_fonts() -> (
    tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]
):
    """Return (SM, MD, LG) font objects local to the calling thread."""
    try:
        return _tls.sm, _tls.md, _tls.lg  # type: ignore[return-value]
    except AttributeError:
        _tls.sm = ImageFont.load_default(size=_FONT_SM_SIZE)
        _tls.md = ImageFont.load_default(size=_FONT_MD_SIZE)
        _tls.lg = ImageFont.load_default(size=_FONT_LG_SIZE)
        return _tls.sm, _tls.md, _tls.lg  # type: ignore[return-value]


def _to_px(gx: int, gy: int) -> tuple[int, int]:
    """Grid cell → pixel coords (y-flip so +y is up), offset by margins."""
    return (
        MARGIN_LEFT + int((FIELD_INCHES - gy * CELL_SIZE) * PPI),
        MARGIN_TOP + int((FIELD_INCHES - gx * CELL_SIZE) * PPI),
    )


def _inches_to_px(x_in: float, y_in: float) -> tuple[float, float]:
    """Inch coordinates → pixel coords (y-flip so +y is up), offset by margins."""
    return (
        MARGIN_LEFT + (FIELD_INCHES - y_in) * PPI,
        MARGIN_TOP + (FIELD_INCHES - x_in) * PPI,
    )


@lru_cache(maxsize=2)
def _render_background(
    blocked_cells: frozenset[tuple[int, int]],
    collision_segments: tuple[tuple[float, float, float, float], ...],
    goal_geometry: tuple[tuple[tuple[float, float], tuple[float, float], int], ...],
    ppi: int = PPI,
    *,
    draw_grid: bool = False,
) -> np.ndarray:
    """Render static field elements, cached as numpy array for thread safety."""
    field_size = FIELD_INCHES * ppi
    img_w = field_size + MARGIN_LEFT + MARGIN_RIGHT
    img_h = field_size + MARGIN_TOP + MARGIN_BOTTOM
    # Round up to multiples of 32 (even-dimension for ffmpeg).
    img_w = (img_w + 31) & ~31
    img_h = (img_h + 31) & ~31
    img = Image.new("P", (img_w, img_h), _P_FIELD)
    img.putpalette(FLAT_PALETTE)
    draw = ImageDraw.Draw(img)

    if draw_grid:
        for g in range(GRID_SIZE + 1):
            _, row_px = _to_px(g, 0)
            col_px, _ = _to_px(0, g)
            draw.line(
                [(MARGIN_LEFT, row_px), (MARGIN_LEFT + field_size, row_px)],
                fill=_P_GRID,
            )
            draw.line(
                [(col_px, MARGIN_TOP), (col_px, MARGIN_TOP + field_size)],
                fill=_P_GRID,
            )

    bdr = 2 * ppi
    draw.rectangle(
        [
            MARGIN_LEFT + bdr,
            MARGIN_TOP + bdr,
            MARGIN_LEFT + field_size - bdr,
            MARGIN_TOP + field_size - bdr,
        ],
        outline=_P_BORDER,
        width=3,
    )

    half = CELL_SIZE * ppi // 2
    for gx, gy in blocked_cells:
        cx, cy = _to_px(gx, gy)
        draw.rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            fill=_P_GRID,
        )

    for ax, ay, bx, by in collision_segments:
        p1 = _inches_to_px(ax, ay)
        p2 = _inches_to_px(bx, by)
        draw.line([p1, p2], fill=_P_COLLISION, width=2)

    for intf_a, intf_b, capacity in goal_geometry:
        px_a = _inches_to_px(*intf_a)
        px_b = _inches_to_px(*intf_b)
        draw.line([px_a, px_b], fill=_P_GOAL, width=GOAL_TUBE_WIDTH)
        slot_r = int(BALL_DIA / 2 * ppi)
        for i in range(capacity):
            t = (i + 0.5) / capacity if capacity > 0 else 0.5
            cx = int(px_a[0] + t * (px_b[0] - px_a[0]))
            cy = int(px_a[1] + t * (px_b[1] - px_a[1]))
            draw.ellipse(
                [cx - slot_r, cy - slot_r, cx + slot_r, cy + slot_r],
                fill=None,
                outline=_P_GOAL_SLOT,
            )

    ox, oy = _to_px(0, 0)
    draw.ellipse([ox - 8, oy - 8, ox + 8, oy + 8], fill=_P_LABEL)
    _, font_md, _ = _get_fonts()
    draw.text((ox + 8, oy - 8), "(0,0)", fill=_P_LABEL, font=font_md)
    ex, ey = _to_px(FIELD_INCHES // CELL_SIZE, 0)
    draw.text((ex + 5, ey - 15), "+x 180°", fill=_P_LABEL, font=font_md)
    yx, yy = _to_px(0, FIELD_INCHES // CELL_SIZE + 1)
    draw.text((yx, yy), "+y 90°", fill=_P_LABEL, font=font_md)

    return np.array(img)


from push_back.env.perf import PerfAccum

_perf = PerfAccum()


def get_render_perf() -> PerfAccum:
    """Return the module-level render PerfAccum (call once after the loop)."""
    return _perf


def render_state(
    state: WorldState,
    ppi: int = PPI,
    *,
    draw_grid: bool = False,
    step: int | None = None,
) -> Image.Image:
    """Return a PIL Image of the current field state."""
    tm = _perf

    with tm.section("bg_hash"):
        blocked = frozenset(state.blocked_cells)
        segments = tuple(
            (seg.ax, seg.ay, seg.bx, seg.by) for seg in state.collision_segments
        )
        goals = tuple(
            (g.interface_a_inches, g.interface_b_inches, g.capacity)
            for g in state.goals
        )

    with tm.section("bg_render"):
        bg_array = _render_background(
            blocked, segments, goals, ppi, draw_grid=draw_grid
        )

    with tm.section("bg_copy"):
        img = Image.fromarray(bg_array, mode="P")
        img.putpalette(FLAT_PALETTE)
        draw = ImageDraw.Draw(img)

    # balls — aggregate by cell
    with tm.section("balls"):
        font_sm, _, _ = _get_fonts()
        ball_r = int(BALL_DIA / 2 * ppi)
        cell_balls: dict[tuple[int, int], dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for row in state.balls_on_field:
            bx, by, color = int(row[0]), int(row[1]), int(row[2])
            cell_balls[(bx, by)][color] += 1
        for (bx, by), colors in cell_balls.items():
            px, py = _to_px(bx, by)
            red_n = colors.get(int(BallColor.RED), 0)
            blue_n = colors.get(int(BallColor.BLUE), 0)
            total = red_n + blue_n
            if total == 1:
                c = BallColor.RED if red_n else BallColor.BLUE
                draw.ellipse(
                    [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
                    fill=_P_BALL[c],
                )
            elif red_n > 0 and blue_n > 0:
                draw.pieslice(
                    [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
                    90,
                    270,
                    fill=_P_BALL_RED,
                )
                draw.pieslice(
                    [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
                    270,
                    90,
                    fill=_P_BALL_BLUE,
                )
                draw.text(
                    (px - ball_r, py - 5), str(red_n), fill=_P_WHITE, font=font_sm
                )
                draw.text((px + 2, py - 5), str(blue_n), fill=_P_WHITE, font=font_sm)
            else:
                c = BallColor.RED if red_n else BallColor.BLUE
                draw.ellipse(
                    [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
                    fill=_P_BALL[c],
                )
                draw.text((px - 4, py - 5), str(total), fill=_P_WHITE, font=font_sm)

    # agents — paste cached P-mode sprites via numpy
    from push_back.env.render_hud import (
        make_agent_sprite,
        render_ball_panel,
        render_robot_panel,
        render_step_panel,
    )

    with tm.section("agents"):
        frame = np.array(img)  # writable copy for numpy pasting
        r = int(AGENT_RADIUS)
        pad = 4
        for i, pose in enumerate(state.agents):
            ci = _P_AGENTS[i] if i < len(_P_AGENTS) else _P_LABEL
            held = state.robot_held_balls[i]
            red_n = sum(1 for b in held if b == BallColor.RED)
            blue_n = sum(1 for b in held if b == BallColor.BLUE)
            sprite, mask = make_agent_sprite(pose.heading, ci, red_n, blue_n)
            sh, sw = sprite.shape
            cx, cy = _to_px(pose.x, pose.y)
            y0, x0 = cy - r - pad, cx - r - pad
            dest = frame[y0 : y0 + sh, x0 : x0 + sw]
            dest[mask] = sprite[mask]

    # HUD — paste cached P-mode panels (black bg, direct copy)
    y_cursor = 4
    with tm.section("hud_step"):
        if step is not None:
            panel = render_step_panel(step)
            ph, pw = panel.shape
            frame[y_cursor : y_cursor + ph, 2 : 2 + pw] = panel
            y_cursor += ph + 2

    with tm.section("hud_robot"):
        positions = tuple((p.x, p.y) for p in state.agents)
        panel = render_robot_panel(positions)
        ph, pw = panel.shape
        frame[y_cursor : y_cursor + ph, 2 : 2 + pw] = panel
        y_cursor += ph + 4

    with tm.section("hud_ball"):
        indexed_balls = tuple(
            sorted(
                (
                    (i, int(row[0]), int(row[1]), int(row[2]))
                    for i, row in enumerate(state.balls_on_field)
                ),
                key=lambda b: (b[3], b[0]),
            ),
        )
        y_cursor += 20
        panel = render_ball_panel(indexed_balls)
        ph, pw = panel.shape
        frame[y_cursor : y_cursor + ph, 2 : 2 + pw] = panel

    with tm.section("to_pil"):
        img = Image.fromarray(frame, "P")
        img.putpalette(FLAT_PALETTE)
    return img
