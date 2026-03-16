"""Field state and constants.

Coordinates use a 4-inch grid with (0, 0) at bottom-left of a 144×144" field.
Heading is discretized to 8 directions (45° increments).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

CELL_SIZE: float = 4.012  # inches per grid cell (35 × 4.012 = 140.42 = FIELD_INTERNAL)
FIELD_INCHES: int = 144
GRID_SIZE: int = 36  # positions 0..35, spanning 35 × CELL_SIZE = FIELD_INTERNAL

ROBOT_DIA: int = 4  # grid cells
ROBOT_RADIUS: int = ROBOT_DIA // 2  # grid cells

# Ball diameters in inches.
# Flat face dia (low-stakes) balls: 3.23"
# Outer dia (high-stakes) balls: 3.84"
BALL_DIA: float = 3.50


def sim_coords(inches: tuple[float, float]) -> tuple[int, int]:
    """Convert an (x, y) position in inches to grid coordinates."""
    return (round(inches[0] / CELL_SIZE), round(inches[1] / CELL_SIZE))


def sim_coords_tiles(inches: tuple[float, float]) -> tuple[int, int]:
    """Convert an (x, y) position in inches to grid coordinates."""
    return (round(inches[0] / CELL_SIZE * 24), round(inches[1] / CELL_SIZE * 24))


class BallColor(IntEnum):
    """Ball alliance color."""

    RED = 1
    BLUE = 2


class Heading(IntEnum):
    """8 compass headings, counterclockwise from East (standard math)."""

    E = 0
    NE = 1
    N = 2
    NW = 3
    W = 4
    SW = 5
    S = 6
    SE = 7


@dataclass
class Pose:
    """Grid position + discrete heading."""

    x: int = 0
    y: int = 0
    heading: int = 0  # Heading enum value 0-7


class Goal:
    """A tube-shaped goal that holds colored balls.

    Each goal has two interface points (a, b) given in inches.
    Capacity is auto-computed from the tube length and BALL_DIA.
    Pushing from side A inserts at index 0 and shifts toward B.
    Pushing from side B inserts at the last index and shifts toward A.
    """

    def __init__(
        self,
        name: str,
        interface_a_inches: tuple[float, float],
        interface_b_inches: tuple[float, float],
    ) -> None:
        self.name = name
        self.interface_a_inches = interface_a_inches
        self.interface_b_inches = interface_b_inches
        self.interface_a = sim_coords(interface_a_inches)
        self.interface_b = sim_coords(interface_b_inches)
        dx = interface_a_inches[0] - interface_b_inches[0]
        dy = interface_a_inches[1] - interface_b_inches[1]
        length = (dx**2 + dy**2) ** 0.5
        self.capacity: int = max(1, int(length / BALL_DIA))
        self.slots: list[BallColor | None] = [None] * self.capacity

    def is_full(self) -> bool:
        return all(s is not None for s in self.slots)

    def count(self) -> int:
        """Number of balls currently in the goal."""
        return sum(1 for s in self.slots if s is not None)

    def push_a(self, color: BallColor) -> BallColor | None:
        """Insert from side A (index 0), shift toward B.

        Returns the ball ejected from side B, or None if a gap absorbed
        the shift.
        """
        ejected = self.slots[-1]
        for j in range(len(self.slots) - 1, 0, -1):
            self.slots[j] = self.slots[j - 1]
        self.slots[0] = color
        return ejected

    def push_b(self, color: BallColor) -> BallColor | None:
        """Insert from side B (last index), shift toward A.

        Returns the ball ejected from side A, or None if a gap absorbed
        the shift.
        """
        ejected = self.slots[0]
        for j in range(len(self.slots) - 1):
            self.slots[j] = self.slots[j + 1]
        self.slots[-1] = color
        return ejected

    def reset(self) -> None:
        self.slots = [None] * self.capacity

    def __repr__(self) -> str:
        contents = [c.name if c else "." for c in self.slots]
        return f"Goal({self.name!r}, [{'|'.join(contents)}])"


def compute_score(goals: list[Goal]) -> tuple[int, int]:
    """Dummy scoring function. Populate with real rules later."""
    # TODO: implement actual scoring combining goal state with other mechanisms
    red = sum(1 for g in goals for s in g.slots if s == BallColor.RED)
    blue = sum(1 for g in goals for s in g.slots if s == BallColor.BLUE)
    return (red, blue)


# Grid-cell deltas (dx, dy) for each heading (E, NE, N, NW, W, SW, S, SE).
HEADING_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 0),  # E
    (1, 1),  # NE
    (0, 1),  # N
    (-1, 1),  # NW
    (-1, 0),  # W
    (-1, -1),  # SW
    (0, -1),  # S
    (1, -1),  # SE
)


def agent_color(agent_index: int) -> BallColor:
    """Return the alliance color for the given agent index (0-3)."""
    return BallColor.RED if agent_index < 2 else BallColor.BLUE


@dataclass
class WorldState:
    """Complete snapshot of the field at one timestep.

    Attributes:
        balls_on_field: (N, 3) int array — columns are (x, y, color).
            color uses BallColor values (1=RED, 2=BLUE).
        agents: length-4 list of Pose for each robot.
        goals: list of Goal objects on the field.
        collision_segments: line segments (in inches) for walls + goal rects.
        blocked_cells: grid cells a robot center cannot occupy.
        robot_held_balls: per-robot list of held BallColor values.
        score: (red, blue) cumulative scores.
        timestep: current simulation tick (0.1 s each).
    """

    balls_on_field: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.int32)
    )
    agents: list[Pose] = field(default_factory=lambda: [Pose() for _ in range(4)])
    goals: list[Goal] = field(default_factory=list)
    collision_segments: list = field(default_factory=list)
    blocked_cells: set = field(default_factory=set)
    robot_held_balls: list[list[BallColor]] = field(
        default_factory=lambda: [[] for _ in range(4)]
    )
    score: tuple[int, int] = (0, 0)
    timestep: int = 0
