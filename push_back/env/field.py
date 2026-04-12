"""Field configuration for the Push-Back game."""

from __future__ import annotations

import math
from typing import NamedTuple

from push_back.env.state import FIELD_INCHES, Goal, sim_coords

FIELD_INTERNAL: float = 140.42  # inches (inside the perimeter)
FIELD_CENTER: float = FIELD_INTERNAL / 2  # 70.21"
SHORT_GOAL_LEN: float = 22.6  # inches, diagonal
_SHORT_OFFSET: float = SHORT_GOAL_LEN / 2 / math.sqrt(2)  # ~7.99"
MATCH_LOADER_WALL_OFFSET: float = 2.58
MATCH_LOADER_LEN: float = 21.34
GOAL_WIDTH_IN: float = 5.53  # physical width of goal tubes in inches

# Default navigation goals for GoToRobot (grid coordinates).
DEFAULT_NAV_GOALS: list[tuple[int, int]] = [
    sim_coords((40, 40)),
    sim_coords((100, 100)),
    sim_coords((40, 100)),
    sim_coords((100, 40)),
]


class CollisionSegment(NamedTuple):
    """A line segment used for collision boundaries, in inches."""

    ax: float
    ay: float
    bx: float
    by: float


def make_default_goals() -> list[Goal]:
    """Create the standard field goals.

    Capacity is auto-computed from tube length / BALL_DIA (3.84").
    Coordinates in inches — Goal discretizes to grid internally.
    """
    return [
        # Long goals
        Goal("long_a", interface_a_inches=(24, 48), interface_b_inches=(24, 96)),
        Goal("long_b", interface_a_inches=(120, 48), interface_b_inches=(120, 96)),
        # Short goals — diagonal, centered on field, 22.6" long
        Goal(
            "short_a",
            interface_a_inches=(
                FIELD_CENTER - _SHORT_OFFSET,
                FIELD_CENTER - _SHORT_OFFSET,
            ),
            interface_b_inches=(
                FIELD_CENTER + _SHORT_OFFSET,
                FIELD_CENTER + _SHORT_OFFSET,
            ),
        ),
        Goal(
            "short_b",
            interface_a_inches=(
                FIELD_CENTER - _SHORT_OFFSET,
                FIELD_CENTER + _SHORT_OFFSET,
            ),
            interface_b_inches=(
                FIELD_CENTER + _SHORT_OFFSET,
                FIELD_CENTER - _SHORT_OFFSET,
            ),
        ),
        # Match loaders
        Goal(
            "loader_blue_0",
            interface_a_inches=(24, MATCH_LOADER_WALL_OFFSET),
            interface_b_inches=(24, MATCH_LOADER_WALL_OFFSET - MATCH_LOADER_LEN),
        ),
        Goal(
            "loader_blue_1",
            interface_a_inches=(24 * 5, MATCH_LOADER_WALL_OFFSET),
            interface_b_inches=(24 * 5, MATCH_LOADER_WALL_OFFSET - MATCH_LOADER_LEN),
        ),
        Goal(
            "loader_red_0",
            interface_a_inches=(24, FIELD_INTERNAL - MATCH_LOADER_WALL_OFFSET),
            interface_b_inches=(
                24,
                FIELD_INTERNAL - MATCH_LOADER_WALL_OFFSET + MATCH_LOADER_LEN,
            ),
        ),
        Goal(
            "loader_red_1",
            interface_a_inches=(24 * 5, FIELD_INTERNAL - MATCH_LOADER_WALL_OFFSET),
            interface_b_inches=(
                24 * 5,
                FIELD_INTERNAL - MATCH_LOADER_WALL_OFFSET + MATCH_LOADER_LEN,
            ),
        ),
    ]


def _goal_rect_segments(goal: Goal) -> list[CollisionSegment]:
    """Return 4 collision segments forming a rectangle around a goal tube."""
    ax_in, ay_in = goal.interface_a_inches
    bx_in, by_in = goal.interface_b_inches
    dx = bx_in - ax_in
    dy = by_in - ay_in
    length = (dx**2 + dy**2) ** 0.5
    if length == 0:
        return []
    perp_x = -dy / length
    perp_y = dx / length
    hw = GOAL_WIDTH_IN / 2
    c = [
        (ax_in + perp_x * hw, ay_in + perp_y * hw),
        (bx_in + perp_x * hw, by_in + perp_y * hw),
        (bx_in - perp_x * hw, by_in - perp_y * hw),
        (ax_in - perp_x * hw, ay_in - perp_y * hw),
    ]
    return [
        CollisionSegment(c[0][0], c[0][1], c[1][0], c[1][1]),
        CollisionSegment(c[1][0], c[1][1], c[2][0], c[2][1]),
        CollisionSegment(c[2][0], c[2][1], c[3][0], c[3][1]),
        CollisionSegment(c[3][0], c[3][1], c[0][0], c[0][1]),
    ]


def make_collision_segments(goals: list[Goal]) -> list[CollisionSegment]:
    """Build all collision segments: field walls + goal rectangles."""
    fi = float(FIELD_INCHES)
    segments: list[CollisionSegment] = [
        CollisionSegment(0, 0, fi, 0),
        CollisionSegment(fi, 0, fi, fi),
        CollisionSegment(fi, fi, 0, fi),
        CollisionSegment(0, fi, 0, 0),
    ]
    for goal in goals:
        segments.extend(_goal_rect_segments(goal))
    # added barrier segments, as verbally pointed out by Hasif in the actual competition
    # note: (ax, ay, bx, by)
    barrier_segments: list[CollisionSegment] = [
        CollisionSegment(20, 50, 30, 50),
        CollisionSegment(30, 50, 30, 90),
        CollisionSegment(30, 90, 20, 90),
        CollisionSegment(20, 90, 20, 50),
    ]
    segments.extend(barrier_segments)
    print(f"Total segments: {len(segments)}")
    return segments
