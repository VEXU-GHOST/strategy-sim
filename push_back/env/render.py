"""Pillow-based renderer for WorldState.

Draws a top-down view of the 144×144" VEX field:
  - gray field with white border
  - colored circles for balls
  - direction indicators for each robot
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from push_back.env.state import (
    BALL_DIA,
    BallColor,
    CELL_SIZE,
    FIELD_INCHES,
    GRID_SIZE,
    ROBOT_RADIUS,
    Goal,
    WorldState,
)

# pixels per inch — controls output resolution
PPI = 5
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
IMG_MARGIN = 25 * PPI  # extra pixels around the field for labels
GRID_COLOR = (100, 100, 100)
COLLISION_SEG_COLOR = (200, 200, 50)


def _to_px(gx: int, gy: int) -> tuple[int, int]:
    """Grid cell → pixel coords (y-flip so +y is up), offset by IMG_MARGIN."""
    return (
        IMG_MARGIN + int((FIELD_INCHES - gy * CELL_SIZE) * PPI),
        IMG_MARGIN + int((FIELD_INCHES - gx * CELL_SIZE) * PPI),
    )


def _inches_to_px(x_in: float, y_in: float) -> tuple[float, float]:
    """Inch coordinates → pixel coords (y-flip so +y is up), offset by IMG_MARGIN."""
    return (
        IMG_MARGIN + (FIELD_INCHES - y_in) * PPI,
        IMG_MARGIN + (FIELD_INCHES - x_in) * PPI,
    )


def _draw_goal(draw: ImageDraw.ImageDraw, goal: Goal, ppi: int) -> None:
    """Draw a goal tube between its two interface points with ball slots."""
    ax, ay = _inches_to_px(*goal.interface_a_inches)
    bx, by = _inches_to_px(*goal.interface_b_inches)

    # tube outline
    draw.line([(ax, ay), (bx, by)], fill=(180, 180, 180), width=GOAL_TUBE_WIDTH)

    # evenly space slots along the tube
    n = goal.capacity
    slot_r = int(BALL_DIA / 2 * ppi)
    for i, slot in enumerate(goal.slots):
        t = (i + 0.5) / n if n > 0 else 0.5
        cx = int(ax + t * (bx - ax))
        cy = int(ay + t * (by - ay))
        if slot is not None:
            fill = BALL_COLORS.get(int(slot), (0, 200, 50))
            draw.ellipse(
                [cx - slot_r, cy - slot_r, cx + slot_r, cy + slot_r],
                fill=fill,
                outline=(255, 255, 255),
            )
        else:
            draw.ellipse(
                [cx - slot_r, cy - slot_r, cx + slot_r, cy + slot_r],
                fill=None,
                outline=(120, 120, 120),
            )


def _draw_agent(
    draw: ImageDraw.ImageDraw,
    gx: int,
    gy: int,
    heading: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a robot as a filled circle with a heading line."""
    cx, cy = _to_px(gx, gy)
    r = AGENT_RADIUS
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="white", width=2)

    # convert discrete heading (0-7, 45° steps) to radians
    heading_rad = heading * (math.pi / 4)
    lx = cx + int(r * math.cos(-heading_rad))  # negate because pixel y is flipped
    ly = cy + int(r * math.sin(-heading_rad))
    draw.line([(cx, cy), (lx, ly)], fill="white", width=3)


def render_state(
    state: WorldState, ppi: int = PPI, *, draw_grid: bool = False
) -> Image.Image:
    """Return a PIL Image of the current field state."""
    field_size = FIELD_INCHES * ppi
    total = field_size + 2 * IMG_MARGIN
    img = Image.new("RGB", (total, total), FIELD_COLOR)
    draw = ImageDraw.Draw(img)

    # grid lines
    if draw_grid:
        for g in range(GRID_SIZE + 1):
            # _to_px returns (col_px, row_px); vary gx for horizontal, gy for vertical
            _, row_px = _to_px(g, 0)
            col_px, _ = _to_px(0, g)
            draw.line(
                [(IMG_MARGIN, row_px), (IMG_MARGIN + field_size, row_px)],
                fill=GRID_COLOR,
            )
            draw.line(
                [(col_px, IMG_MARGIN), (col_px, IMG_MARGIN + field_size)],
                fill=GRID_COLOR,
            )

    # field border
    bdr = 2 * ppi
    draw.rectangle(
        [
            IMG_MARGIN + bdr,
            IMG_MARGIN + bdr,
            IMG_MARGIN + field_size - bdr,
            IMG_MARGIN + field_size - bdr,
        ],
        outline=BORDER_COLOR,
        width=3,
    )

    # origin marker and axes
    ox, oy = _to_px(0, 0)  # pixel position of grid (0,0)
    font = ImageFont.load_default()
    # origin dot + label
    draw.ellipse([ox - 8, oy - 8, ox + 8, oy + 8], fill=LABEL_COLOR)
    draw.text((ox + 8, oy - 8), "(0,0)", fill=LABEL_COLOR, font=font)
    # +x axis labels at far end (144" + 1 grid step)
    draw.text(
        _to_px(FIELD_INCHES // CELL_SIZE, 0), "+x 180°", fill=LABEL_COLOR, font=font
    )
    # +y axis labels at far end (144" + 1 grid step)
    draw.text(
        _to_px(0, FIELD_INCHES // CELL_SIZE + 1), "+y 90°", fill=LABEL_COLOR, font=font
    )

    # blocked cells
    half = CELL_SIZE * ppi // 2
    for gx, gy in state.blocked_cells:
        cx, cy = _to_px(gx, gy)
        draw.rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            fill=GRID_COLOR,
        )

    # collision segments (walls + goal boundaries)
    for seg in state.collision_segments:
        p1 = _inches_to_px(seg.ax, seg.ay)
        p2 = _inches_to_px(seg.bx, seg.by)
        draw.line([p1, p2], fill=COLLISION_SEG_COLOR, width=2)

    # goal ball slots
    for goal in state.goals:
        _draw_goal(draw, goal, ppi)

    # balls
    ball_r = int(BALL_DIA / 2 * ppi)
    for row in state.balls_on_field:
        bx, by, color = int(row[0]), int(row[1]), int(row[2])
        px, py = _to_px(bx, by)
        fill = BALL_COLORS.get(color, (0, 200, 50))
        draw.ellipse(
            [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
            fill=fill,
        )

    # agents
    for i, pose in enumerate(state.agents):
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (200, 200, 200)
        _draw_agent(draw, pose.x, pose.y, pose.heading, color)

    return img
