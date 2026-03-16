"""Pillow-based renderer for WorldState.

Draws a top-down view of the 144×144" VEX field:
  - gray field with white border
  - colored circles for balls
  - direction indicators for each robot
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from push_back.env.state import (
    BALL_DIA,
    BallColor,
    CELL_SIZE,
    FIELD_INCHES,
    GRID_SIZE,
    HEADING_DELTAS,
    ROBOT_RADIUS,
    WorldState,
)

# pixels per inch — controls output resolution
PPI = 3
IMG_SIZE = FIELD_INCHES * PPI  # 720 px

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
AXIS_COLOR = (140, 140, 140)
AXIS_LEN = 30 * PPI  # length of axis arrows in pixels
# per-side margins (pixels) — only as wide as the content on that edge
MARGIN_LEFT = 24 * PPI  # room for HUD sidebar
MARGIN_TOP = 8 * PPI
MARGIN_RIGHT = 20 * PPI  # room for origin (0,0) label
MARGIN_BOTTOM = 8 * PPI
GRID_COLOR = (100, 100, 100)
COLLISION_SEG_COLOR = (200, 200, 50)

# cached fonts — scaled to PPI (sizes tuned at PPI=5, scale linearly)
_FONT_SM: ImageFont.FreeTypeFont = ImageFont.load_default(
    size=max(8, int(12 * PPI / 5))
)
_FONT_MD: ImageFont.FreeTypeFont = ImageFont.load_default(
    size=max(10, int(20 * PPI / 5))
)
_FONT_LG: ImageFont.FreeTypeFont = ImageFont.load_default(
    size=max(12, int(34 * PPI / 5))
)


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


@lru_cache(maxsize=4)
def _make_agent_sprite(
    heading: int,
    color: tuple[int, int, int],
    red_n: int,
    blue_n: int,
) -> Image.Image:
    """Return an RGBA sprite for a robot, cached by visual state."""
    r = int(AGENT_RADIUS)
    pad = 4  # extra pixels for stroke overflow
    size = 2 * r + 2 * pad
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = r + pad, r + pad
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="white", width=2)

    dx, dy = HEADING_DELTAS[heading]
    length = math.hypot(dx, dy) or 1.0
    lx = cx + int(r * (-dy) / length)
    ly = cy + int(r * (-dx) / length)
    draw.line([(cx, cy), (lx, ly)], fill="white", width=3)

    red_str = str(red_n)
    blue_str = str(blue_n)
    spacing = 4
    rw = _FONT_LG.getlength(red_str)
    bw = _FONT_LG.getlength(blue_str)
    ascent, descent = _FONT_LG.getmetrics()
    text_h = ascent + descent
    total_w = rw + spacing + bw
    rx = cx - total_w / 2
    ty = cy - text_h / 2
    bx_pos = rx + rw + spacing
    draw.text(
        (rx, ty),
        red_str,
        fill=BALL_COLORS[BallColor.RED],
        font=_FONT_LG,
        stroke_width=2,
        stroke_fill="black",
    )
    draw.text(
        (bx_pos, ty),
        blue_str,
        fill=BALL_COLORS[BallColor.BLUE],
        font=_FONT_LG,
        stroke_width=2,
        stroke_fill="black",
    )
    return img


@lru_cache(maxsize=2)
def _render_background(
    blocked_cells: frozenset[tuple[int, int]],
    collision_segments: tuple[tuple[float, float, float, float], ...],
    goal_geometry: tuple[tuple[tuple[float, float], tuple[float, float], int], ...],
    ppi: int = PPI,
    *,
    draw_grid: bool = False,
) -> Image.Image:
    """Render static field elements that are identical every frame."""
    field_size = FIELD_INCHES * ppi
    img_w = field_size + MARGIN_LEFT + MARGIN_RIGHT
    img_h = field_size + MARGIN_TOP + MARGIN_BOTTOM
    # Round up to multiples of 32 so each RGB24 row is SIMD-aligned (AVX).
    # This also satisfies ffmpeg's even-dimension requirement.
    img_w = (img_w + 31) & ~31
    img_h = (img_h + 31) & ~31
    img = Image.new("RGB", (img_w, img_h), FIELD_COLOR)
    draw = ImageDraw.Draw(img)

    if draw_grid:
        for g in range(GRID_SIZE + 1):
            _, row_px = _to_px(g, 0)
            col_px, _ = _to_px(0, g)
            draw.line(
                [(MARGIN_LEFT, row_px), (MARGIN_LEFT + field_size, row_px)],
                fill=GRID_COLOR,
            )
            draw.line(
                [(col_px, MARGIN_TOP), (col_px, MARGIN_TOP + field_size)],
                fill=GRID_COLOR,
            )

    bdr = 2 * ppi
    draw.rectangle(
        [
            MARGIN_LEFT + bdr,
            MARGIN_TOP + bdr,
            MARGIN_LEFT + field_size - bdr,
            MARGIN_TOP + field_size - bdr,
        ],
        outline=BORDER_COLOR,
        width=3,
    )

    half = CELL_SIZE * ppi // 2
    for gx, gy in blocked_cells:
        cx, cy = _to_px(gx, gy)
        draw.rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            fill=GRID_COLOR,
        )

    for ax, ay, bx, by in collision_segments:
        p1 = _inches_to_px(ax, ay)
        p2 = _inches_to_px(bx, by)
        draw.line([p1, p2], fill=COLLISION_SEG_COLOR, width=2)

    for intf_a, intf_b, capacity in goal_geometry:
        px_a = _inches_to_px(*intf_a)
        px_b = _inches_to_px(*intf_b)
        draw.line([px_a, px_b], fill=(180, 180, 180), width=GOAL_TUBE_WIDTH)
        slot_r = int(BALL_DIA / 2 * ppi)
        for i in range(capacity):
            t = (i + 0.5) / capacity if capacity > 0 else 0.5
            cx = int(px_a[0] + t * (px_b[0] - px_a[0]))
            cy = int(px_a[1] + t * (px_b[1] - px_a[1]))
            draw.ellipse(
                [cx - slot_r, cy - slot_r, cx + slot_r, cy + slot_r],
                fill=None,
                outline=(120, 120, 120),
            )

    ox, oy = _to_px(0, 0)
    draw.ellipse([ox - 8, oy - 8, ox + 8, oy + 8], fill=LABEL_COLOR)
    draw.text((ox + 8, oy - 8), "(0,0)", fill=LABEL_COLOR, font=_FONT_MD)
    ex, ey = _to_px(FIELD_INCHES // CELL_SIZE, 0)
    draw.text((ex + 5, ey - 15), "+x 180°", fill=LABEL_COLOR, font=_FONT_MD)
    yx, yy = _to_px(0, FIELD_INCHES // CELL_SIZE + 1)
    draw.text((yx, yy), "+y 90°", fill=LABEL_COLOR, font=_FONT_MD)

    return img


# Per-phase timing accumulators (seconds).
_timings: dict[str, float] = {
    "bg_hash": 0.0,
    "bg_copy": 0.0,
    "balls": 0.0,
    "agents": 0.0,
    "hud_step_render": 0.0,
    "hud_step_paste": 0.0,
    "hud_robot_render": 0.0,
    "hud_robot_paste": 0.0,
    "hud_ball_render": 0.0,
    "hud_ball_paste": 0.0,
    "tobytes": 0.0,
}
_frame_count: int = 0


def get_render_timings() -> dict[str, float]:
    """Return accumulated render phase timings and reset them."""
    global _frame_count
    result = {k: v for k, v in _timings.items()}
    result["frames"] = float(_frame_count)
    for k in _timings:
        _timings[k] = 0.0
    _frame_count = 0
    return result


def render_state(
    state: WorldState,
    ppi: int = PPI,
    *,
    draw_grid: bool = False,
    step: int | None = None,
) -> Image.Image:
    """Return a PIL Image of the current field state."""
    global _frame_count
    _frame_count += 1

    t = time.perf_counter()
    blocked = frozenset(state.blocked_cells)
    segments = tuple(
        (seg.ax, seg.ay, seg.bx, seg.by) for seg in state.collision_segments
    )
    goals = tuple(
        (g.interface_a_inches, g.interface_b_inches, g.capacity) for g in state.goals
    )
    _timings["bg_hash"] += time.perf_counter() - t

    t = time.perf_counter()
    bg = _render_background(blocked, segments, goals, ppi, draw_grid=draw_grid)
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    _timings["bg_copy"] += time.perf_counter() - t

    # balls — aggregate by cell
    t = time.perf_counter()
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
                fill=BALL_COLORS[c],
            )
        elif red_n > 0 and blue_n > 0:
            # Mixed — left half red, right half blue
            draw.pieslice(
                [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
                90,
                270,
                fill=BALL_COLORS[BallColor.RED],
            )
            draw.pieslice(
                [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
                270,
                90,
                fill=BALL_COLORS[BallColor.BLUE],
            )
            draw.text((px - ball_r, py - 5), str(red_n), fill="white", font=_FONT_SM)
            draw.text((px + 2, py - 5), str(blue_n), fill="white", font=_FONT_SM)
        else:
            # Single color, multiple balls
            c = BallColor.RED if red_n else BallColor.BLUE
            draw.ellipse(
                [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
                fill=BALL_COLORS[c],
            )
            draw.text((px - 4, py - 5), str(total), fill="white", font=_FONT_SM)
    _timings["balls"] += time.perf_counter() - t

    # agents — paste cached sprites
    t = time.perf_counter()
    r = int(AGENT_RADIUS)
    pad = 4
    for i, pose in enumerate(state.agents):
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (200, 200, 200)
        held = state.robot_held_balls[i]
        red_n = sum(1 for b in held if b == BallColor.RED)
        blue_n = sum(1 for b in held if b == BallColor.BLUE)
        sprite = _make_agent_sprite(pose.heading, color, red_n, blue_n)
        cx, cy = _to_px(pose.x, pose.y)
        img.paste(sprite, (cx - r - pad, cy - r - pad), sprite)
    _timings["agents"] += time.perf_counter() - t

    # HUD panels — cached RGBA overlays
    from push_back.env.render_hud import (
        render_ball_panel,
        render_robot_panel,
        render_step_panel,
    )

    y_cursor = 4
    t = time.perf_counter()
    if step is not None:
        step_img = render_step_panel(step)
        _timings["hud_step_render"] += time.perf_counter() - t
        t = time.perf_counter()
        img.paste(step_img, (2, y_cursor), step_img)
        y_cursor += step_img.height + 2
    _timings["hud_step_paste"] += time.perf_counter() - t

    t = time.perf_counter()
    positions = tuple((p.x, p.y) for p in state.agents)
    robot_img = render_robot_panel(positions)
    _timings["hud_robot_render"] += time.perf_counter() - t
    t = time.perf_counter()
    img.paste(robot_img, (2, y_cursor), robot_img)
    y_cursor += robot_img.height + 4
    _timings["hud_robot_paste"] += time.perf_counter() - t

    t = time.perf_counter()
    indexed_balls = tuple(
        sorted(
            (
                (i, int(row[0]), int(row[1]), int(row[2]))
                for i, row in enumerate(state.balls_on_field)
            ),
            key=lambda b: (b[3], b[0]),
        )
    )
    y_cursor += 20
    ball_img = render_ball_panel(indexed_balls)
    _timings["hud_ball_render"] += time.perf_counter() - t
    t = time.perf_counter()
    img.paste(ball_img, (2, y_cursor), ball_img)
    _timings["hud_ball_paste"] += time.perf_counter() - t

    return img
