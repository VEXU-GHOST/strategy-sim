"""Grid pathfinder: converts a target (x, y) into a sequence of primitives.

Uses simple greedy heading alignment then straight-line movement.
Good enough for a 36×36 grid with no obstacles (agents don't block each other
yet). Swap in A* later if collision avoidance is needed.
"""

from __future__ import annotations

from collections import deque

from push_back.env.state import (
    GRID_SIZE,
    Heading,
    Pose,
)
from push_back.env.robots.random_robot import Action, HEADING_DELTAS


def _desired_heading(dx: int, dy: int) -> int | None:
    """Return the Heading index for the direction (dx, dy), or None if (0,0)."""
    for h, (hx, hy) in enumerate(HEADING_DELTAS):
        if hx == dx and hy == dy:
            return h
    return None


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def plan(start: Pose, goal_x: int, goal_y: int, turn_ticks: int = 3) -> deque[int]:
    """Return a deque of primitive Action ints to move *start* to (goal_x, goal_y).

    Strategy: step axis-aligned (or diagonally when both axes need movement),
    aligning heading first.  Each 45° heading change costs *turn_ticks* ticks.
    """
    goal_x = max(0, min(goal_x, GRID_SIZE - 1))
    goal_y = max(0, min(goal_y, GRID_SIZE - 1))

    actions: deque[int] = deque()
    x, y, heading = start.x, start.y, start.heading

    while x != goal_x or y != goal_y:
        dx = _sign(goal_x - x)
        dy = _sign(goal_y - y)
        target_h = _desired_heading(dx, dy)
        if target_h is None:
            break  # already at goal

        # turn to face target
        while heading != target_h:
            # shortest turn direction
            diff = (target_h - heading) % 8
            if diff <= 4:
                for _ in range(turn_ticks):
                    actions.append(Action.TURN_LEFT)
                heading = (heading + 1) % 8
            else:
                for _ in range(turn_ticks):
                    actions.append(Action.TURN_RIGHT)
                heading = (heading - 1) % 8

        # move forward one cell
        actions.append(Action.FORWARD)
        x += dx
        y += dy

    return actions
