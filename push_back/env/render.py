"""Pillow-based renderer for WorldState.

Draws a top-down view of the 144×144" VEX field:
  - gray field with white border
  - colored circles for balls
  - direction indicators for each robot
"""

from __future__ import annotations

import math
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from push_back.env.state import (
    BALL_DIA,
    BallColor,
    CELL_SIZE,
    FIELD_INCHES,
    GRID_SIZE,
    HEADING_DELTAS,
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
    held_balls: list[int],
) -> None:
    """Draw a robot as a filled circle with heading line and held-ball counts."""
    cx, cy = _to_px(gx, gy)
    r = AGENT_RADIUS
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="white", width=2)

    # heading line — pixel axes are swapped & flipped vs grid axes:
    #   pixel_x ← -(grid_y),  pixel_y ← -(grid_x)
    dx, dy = HEADING_DELTAS[heading]
    length = math.hypot(dx, dy) or 1.0
    lx = cx + int(r * (-dy) / length)
    ly = cy + int(r * (-dx) / length)
    draw.line([(cx, cy), (lx, ly)], fill="white", width=3)

    # held ball counts
    red_n = sum(1 for b in held_balls if b == BallColor.RED)
    blue_n = sum(1 for b in held_balls if b == BallColor.BLUE)
    font = ImageFont.load_default()
    if red_n > 0:
        draw.text(
            (cx - r + 2, cy + int(r * 0.3)),
            str(red_n),
            fill=BALL_COLORS[BallColor.RED],
            font=font,
        )
    if blue_n > 0:
        draw.text(
            (cx + int(r * 0.3), cy + int(r * 0.3)),
            str(blue_n),
            fill=BALL_COLORS[BallColor.BLUE],
            font=font,
        )


def render_state(
    state: WorldState,
    ppi: int = PPI,
    *,
    draw_grid: bool = False,
    step: int | None = None,
) -> Image.Image:
    """Return a PIL Image of the current field state."""
    field_size = FIELD_INCHES * ppi
    total = field_size + 2 * IMG_MARGIN
    # h264 yuv420p needs even dimensions
    if total % 2:
        total += 1
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
    try:
        label_font = ImageFont.load_default(size=20)
    except TypeError:
        label_font = font

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

    # balls — aggregate by cell
    ball_r = int(BALL_DIA / 2 * ppi)
    cell_balls: dict[tuple[int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in state.balls_on_field:
        bx, by, color = int(row[0]), int(row[1]), int(row[2])
        cell_balls[(bx, by)][color] += 1

    font = ImageFont.load_default()
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
            draw.text((px - ball_r, py - 5), str(red_n), fill="white", font=font)
            draw.text((px + 2, py - 5), str(blue_n), fill="white", font=font)
        else:
            # Single color, multiple balls
            c = BallColor.RED if red_n else BallColor.BLUE
            draw.ellipse(
                [px - ball_r, py - ball_r, px + ball_r, py + ball_r],
                fill=BALL_COLORS[c],
            )
            draw.text((px - 4, py - 5), str(total), fill="white", font=font)

    # agents
    for i, pose in enumerate(state.agents):
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (200, 200, 200)
        held = [int(b) for b in state.robot_held_balls[i]]
        _draw_agent(draw, pose.x, pose.y, pose.heading, color, held)

    # origin dot + label (drawn last so they're on top)
    draw.ellipse([ox - 8, oy - 8, ox + 8, oy + 8], fill=LABEL_COLOR)
    draw.text((ox + 8, oy - 8), "(0,0)", fill=LABEL_COLOR, font=label_font)
    # +x axis label at far end
    ex, ey = _to_px(FIELD_INCHES // CELL_SIZE, 0)
    draw.text((ex, ey - 25), "+x 180°", fill=LABEL_COLOR, font=label_font)
    # +y axis label at far end
    yx, yy = _to_px(0, FIELD_INCHES // CELL_SIZE + 1)
    draw.text((yx, yy - 25), "+y 90°", fill=LABEL_COLOR, font=label_font)

    # step number (top-left corner)
    if step is not None:
        bbox = label_font.getbbox(f"Step {step}")
        draw.rectangle(
            [8, 8, 12 + bbox[2], 12 + bbox[3]],
            fill=(0, 0, 0),
        )
        draw.text((10, 10), f"Step {step}", fill=LABEL_COLOR, font=label_font)

    # robot locations (below step number)
    y_start = 35 if step is not None else 10
    line_height = 22
    # background for all robot labels
    max_label = max(
        (
            label_font.getbbox(f"R{i}: ({p.x}, {p.y})")[2]
            for i, p in enumerate(state.agents)
        ),
        default=0,
    )
    bg_bottom = y_start + len(state.agents) * line_height + 2
    draw.rectangle([8, y_start - 2, 12 + max_label, bg_bottom], fill=(0, 0, 0))
    for i, pose in enumerate(state.agents):
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (200, 200, 200)
        label = f"R{i}: ({pose.x}, {pose.y})"
        draw.text((10, y_start + i * line_height), label, fill=color, font=label_font)

    # ball locations (below robots, separated by a blank line)
    ball_y = bg_bottom + line_height
    # sort: red balls first then blue, each group by original index
    indexed_balls = [
        (i, int(row[0]), int(row[1]), int(row[2]))
        for i, row in enumerate(state.balls_on_field)
    ]
    indexed_balls.sort(key=lambda b: (b[3], b[0]))
    ball_labels: list[tuple[str, tuple[int, int, int]]] = []
    for idx, bx, by, bc in indexed_balls:
        c = BALL_COLORS.get(bc, (200, 200, 200))
        ball_labels.append((f"B{idx}: ({bx}, {by})", c))
    if ball_labels:
        max_ball_w = max(label_font.getbbox(lbl)[2] for lbl, _ in ball_labels)
        ball_bg_bottom = ball_y + len(ball_labels) * line_height + 2
        draw.rectangle([8, ball_y - 2, 12 + max_ball_w, ball_bg_bottom], fill=(0, 0, 0))
        for j, (lbl, c) in enumerate(ball_labels):
            draw.text((10, ball_y + j * line_height), lbl, fill=c, font=label_font)

    return img
