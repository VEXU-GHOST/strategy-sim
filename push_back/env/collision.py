"""Collision detection for the Push-Back simulation.

Provides:
- ``compute_blocked_cells`` — precompute grid cells a robot center cannot occupy.
- ``resolve_moves`` — reject proposed poses that collide with obstacles or
  other robots.
"""

from __future__ import annotations

import math

from push_back.env.field import CollisionSegment # modified import to include barrier segments
from push_back.env.state import CELL_SIZE, GRID_SIZE, Pose


def _point_seg_dist(px: float, py: float, seg: CollisionSegment) -> float:
    """Minimum distance from point (px, py) to a line segment."""
    ax, ay, bx, by = seg
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def compute_blocked_cells(
    segments: list[CollisionSegment], robot_radius_cells: int
) -> set[tuple[int, int]]:
    """Return the set of grid cells where a robot center cannot be placed.

    For each grid cell, convert its center to inches and check if it's
    within ``robot_radius_cells * CELL_SIZE`` of any collision segment.
    """
    margin_in = robot_radius_cells * CELL_SIZE
    blocked: set[tuple[int, int]] = set()
    for gx in range(GRID_SIZE):
        cx_in = gx * CELL_SIZE
        for gy in range(GRID_SIZE):
            cy_in = gy * CELL_SIZE
            for seg in segments:
                if _point_seg_dist(cx_in, cy_in, seg) < margin_in:
                    blocked.add((gx, gy))
                    break
    return blocked


def is_cell_blocked(
    gx: int,
    gy: int,
    blocked_cells: set[tuple[int, int]],
) -> bool:
    """Check if a single grid cell is blocked for a robot center."""
    if gx < 0 or gx >= GRID_SIZE or gy < 0 or gy >= GRID_SIZE:
        return True
    return (gx, gy) in blocked_cells


def _robots_overlap(a: Pose, b: Pose, robot_radius_cells: int) -> bool:
    """True if two robots' circular footprints overlap on the grid."""
    dx = a.x - b.x
    dy = a.y - b.y
    # Two circles overlap when center distance < sum of radii.
    # Both robots have the same radius so threshold = diameter.
    return (dx * dx + dy * dy) < (2 * robot_radius_cells) ** 2


def resolve_moves(
    current: list[Pose],
    proposed: list[Pose],
    blocked_cells: set[tuple[int, int]],
    robot_radius_cells: int,
) -> list[Pose]:
    """Validate proposed poses against obstacles and robot-robot collisions.

    1. Any proposed pose in ``blocked_cells`` is rejected (robot stays put).
    2. For each pair of robots, if their proposed poses overlap, **both**
       revert to their current pose.

    Returns the final list of poses (same length as *current*).
    """
    n = len(current)
    result = list(proposed)

    # Phase 1: reject moves into blocked cells
    for i in range(n):
        if is_cell_blocked(result[i].x, result[i].y, blocked_cells):
            result[i] = Pose(current[i].x, current[i].y, proposed[i].heading)

    # Phase 2: robot-robot overlap — both revert position
    reverted: set[int] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if _robots_overlap(result[i], result[j], robot_radius_cells):
                reverted.add(i)
                reverted.add(j)
    for i in reverted:
        result[i] = Pose(current[i].x, current[i].y, proposed[i].heading)

    return result
