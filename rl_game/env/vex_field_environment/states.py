"""The world data: FieldState owns four RobotStates, objects and goal stacks.

This file knows coordinates, inventory and prototype points. It does not select
actions, generate routes or decide what a robot can observe.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .actions import Action

EPS = 1e-9
RED, BLUE = 0, 1
COLORS = {"red": RED, "blue": BLUE}
ITEM_TYPES = ("blue_yellow", "red_yellow", "red_blue", "yellow_yellow", "cup")
PIN_FACES = (("blue", "yellow"), ("red", "yellow"), ("red", "blue"), ("yellow", "yellow"))
PALETTE = {"#20B7E8": "blue", "#EC0C3A": "red", "#E4003B": "red", "#FFFF00": "yellow"}
DEFAULTS = {
    "cells_per_foot": 12, "seconds_per_step": 0.25, "episode_seconds": 120.0,
    "robot_radius": 0.45, "robot_speed": 3.0, "turn_speed": 180.0,
    "interaction_reach": 0.95, "goal_radius": 0.27, "goal_capacity": 10,
    "alternate_pin_and_cup": True, "colored_face_points": 5.0, "yellow_face_points": 10.0,
    "collect_seconds": 0.8, "score_seconds": 1.0, "toggle_seconds": 0.7,
    "matchload_seconds": 0.8, "drop_seconds": 0.25, "park_seconds": 0.5,
    "wait_seconds": 0.25, "sensor_range": 4.0, "sensor_fov_degrees": 90.0,
}


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float = 0.0  # Degrees, 0=east, positive counterclockwise.

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class PendingTask:
    action: Action
    poses: tuple[Pose, ...]  # Start pose followed by one pose per movement tick.
    interaction_remaining: float
    started_at: float
    estimated_duration: float
    path_index: int = 0

    @property
    def moving(self) -> bool:
        return self.path_index + 1 < len(self.poses)

    def remaining_seconds(self, tick: float) -> float:
        movement = max(0, len(self.poses) - self.path_index - 1) * tick
        interaction = math.ceil(max(0.0, self.interaction_remaining - EPS) / tick) * tick
        return movement + interaction


@dataclass
class RobotState:
    agent_id: str
    team: int
    row: float
    col: float
    heading: float = 0.0
    held_items: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=int))
    task: PendingTask | None = None
    parked: bool = False


@dataclass
class Goal:
    goal_id: str
    row: int
    col: int
    max_items: int
    quadrant: int | None
    stack: list[tuple[int, str | None]] = field(default_factory=list)


@dataclass
class LoaderState:
    loader_id: str
    position: tuple[float, float]
    team: int | None
    stock: np.ndarray


class FieldState:
    """Mutable source of truth; clone() makes an independent search world."""

    def __init__(self, config: dict):
        self.config = deepcopy(config)
        self.settings = {**DEFAULTS, **config.get("environment", {})}
        self.half_width = float(config["field"]["half_width"])
        self.half_height = float(config["field"]["half_height"])
        if not math.isfinite(self.half_width) or self.half_width != self.half_height:
            raise ValueError("Use a finite square field for the original V1 bridge")
        for key, value in self.settings.items():
            if isinstance(DEFAULTS.get(key), bool) or key not in DEFAULTS:
                continue
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("cells_per_foot", "goal_capacity"):
            if int(self.settings[key]) != self.settings[key]:
                raise ValueError(f"{key} must be an integer")
        if self.settings["sensor_fov_degrees"] > 360:
            raise ValueError("sensor_fov_degrees must be at most 360")
        if self.half_width <= self.settings["robot_radius"]:
            raise ValueError("Field must be wider than the robot")
        self.cells_per_foot = int(self.settings["cells_per_foot"])
        self.board_size = round(2 * self.half_width * self.cells_per_foot) + 1
        self.board = np.zeros((5, self.board_size, self.board_size), dtype=np.int32)
        self.time = 0.0
        self.step_count = 0
        self.zone_colors = np.zeros(4, dtype=int)  # 0 neutral, 1 red, 2 blue.
        layout = config["layout"]

        for point, colors in layout.get("pins", []):
            faces = tuple(PALETTE.get(c.upper(), c.lower()) for c in colors)
            channel = next((i for i, pair in enumerate(PIN_FACES) if sorted(pair) == sorted(faces)), None)
            if channel is None:
                raise ValueError(f"Unsupported Pin colors: {colors}")
            row, col = self.world_to_cell(point)
            self.board[channel, row, col] += 1
        for point in layout.get("gray_cups", []) + layout.get("clear_cups", []):
            row, col = self.world_to_cell(point)
            self.board[4, row, col] += 1

        self.goals: dict[str, Goal] = {}
        for index, entry in enumerate(layout.get("goals", [])):
            options = entry if isinstance(entry, dict) else {}
            point = options["position"] if options else entry[0]
            row, col = self.world_to_cell(point)
            quadrant = None if math.hypot(*point) < EPS else self.quadrant(row, col)
            goal = Goal(options.get("id", f"goal_{index}"), row, col,
                        int(options.get("capacity", self.settings["goal_capacity"])),
                        options.get("quadrant", quadrant))
            if goal.goal_id in self.goals or goal.max_items < 1:
                raise ValueError("Goal IDs must be unique and capacities positive")
            if goal.quadrant is not None and goal.quadrant not in range(4):
                raise ValueError("Goal quadrant must be 0..3 or null")
            self.goals[goal.goal_id] = goal

        self.toggles = self._keyed_entries(layout.get("toggles", []))
        for toggle in self.toggles.values():
            self.world_to_cell(toggle["position"])
            if toggle["quadrant"] not in range(4):
                raise ValueError("Toggle quadrant must be 0..3")
        self.loaders: dict[str, LoaderState] = {}
        for entry in layout.get("matchloaders", []):
            position = self.cell_to_world(*self.world_to_cell(entry["position"]))
            stock_data = entry.get("stock", {})
            if any(item not in ITEM_TYPES for item in stock_data):
                raise ValueError("Unknown matchloader stock item")
            values = [stock_data.get(item, 0) for item in ITEM_TYPES]
            if any(int(v) != v or v < 0 for v in values):
                raise ValueError("Matchloader stock must be nonnegative integers")
            if entry["id"] in self.loaders:
                raise ValueError("Matchloader IDs must be unique")
            team = COLORS[entry["team"]] if entry.get("team") else None
            self.loaders[entry["id"]] = LoaderState(entry["id"], position, team, np.array(values, dtype=int))
        self.parking_zones = self._keyed_entries(layout.get("parking_zones", []))
        for zone in self.parking_zones.values():
            xmin, xmax, ymin, ymax = map(float, zone["bounds"])
            self.world_to_cell((xmin, ymin))
            self.world_to_cell((xmax, ymax))
            if xmax - xmin < 2 * self.settings["robot_radius"] or ymax - ymin < 2 * self.settings["robot_radius"]:
                raise ValueError("Parking zone must fit the robot body")
            if zone.get("team") is not None and zone["team"] not in COLORS:
                raise ValueError("Parking zone team must be red, blue or null")

        default_spawns = [
            {"id": "A0", "team": "blue", "position": config.get("robot", {}).get("start", [-3.9, 3.9])},
            {"id": "A1", "team": "blue", "position": [-4.8, -4.8]},
            {"id": "B0", "team": "red", "position": [3.9, -3.9]},
            {"id": "B1", "team": "red", "position": [4.8, 4.8]},
        ]
        self.robots: dict[str, RobotState] = {}
        for entry in config.get("robots", default_spawns):
            row, col = self.world_to_cell(entry["position"])
            if entry["id"] in self.robots:
                raise ValueError("Robot IDs must be unique")
            heading = float(entry.get("heading", 0.0))
            if not math.isfinite(heading):
                raise ValueError("Robot heading must be finite")
            self.robots[entry["id"]] = RobotState(entry["id"], COLORS[entry["team"]], row, col, heading % 360)

    @staticmethod
    def _keyed_entries(entries: list[dict]) -> dict[str, dict]:
        result = {}
        for entry in entries:
            if entry["id"] in result:
                raise ValueError("Layout IDs must be unique within each category")
            result[entry["id"]] = deepcopy(entry)
        return result

    def world_to_cell(self, position) -> tuple[int, int]:
        x, y = map(float, position)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Position must be finite")
        if not (-self.half_width <= x <= self.half_width and -self.half_height <= y <= self.half_height):
            raise ValueError("Position is outside the field")
        return round((self.half_height - y) * self.cells_per_foot), round((x + self.half_width) * self.cells_per_foot)

    def cell_to_world(self, row: float, col: float) -> tuple[float, float]:
        return col / self.cells_per_foot - self.half_width, self.half_height - row / self.cells_per_foot

    def pose(self, robot_id: str) -> Pose:
        robot = self.robots[robot_id]
        return Pose(*self.cell_to_world(robot.row, robot.col), robot.heading)

    def set_pose(self, robot_id: str, pose: Pose) -> None:
        # Robot poses are continuous; only floor-object coordinates are rounded.
        robot = self.robots[robot_id]
        robot.row = (self.half_height - pose.y) * self.cells_per_foot
        robot.col = (pose.x + self.half_width) * self.cells_per_foot
        robot.heading = pose.theta % 360

    def quadrant(self, row: float, col: float) -> int:
        half = self.board_size / 2
        return int(row >= half) * 2 + int(col >= half)

    def object_states(self) -> list[dict]:
        return [{"object_id": f"obj_c{ch}_r{r}_c{c}", "channel": int(ch),
                 "item": ITEM_TYPES[ch], "row": int(r), "col": int(c),
                 "position": self.cell_to_world(int(r), int(c)),
                 "region": f"q{self.quadrant(int(r), int(c))}", "count": int(self.board[ch, r, c])}
                for ch, r, c in np.argwhere(self.board > 0)]

    def find_object(self, object_id: str) -> dict:
        result = next((obj for obj in self.object_states() if obj["object_id"] == object_id), None)
        if result is None:
            raise ValueError("object_missing_or_already_collected")
        return result

    def goal_states(self) -> np.ndarray:
        width = max((g.max_items for g in self.goals.values()), default=0)
        result = np.zeros((len(self.goals), width), dtype=int)
        for i, goal in enumerate(self.goals.values()):
            result[i, :len(goal.stack)] = [ch + 1 for ch, _ in goal.stack]
        return result

    def loader_stock(self, team: int) -> np.ndarray:
        result = np.zeros(5, dtype=int)
        for loader in self.loaders.values():
            if loader.team == team:
                result += loader.stock
        return result

    def scores(self) -> list[float]:
        result = [0.0, 0.0]
        for goal in self.goals.values():
            for channel, face in goal.stack:
                if channel == 4:
                    continue
                if face in COLORS:
                    result[COLORS[face]] += self.settings["colored_face_points"]
                elif face == "yellow" and goal.quadrant is not None:
                    owner = int(self.zone_colors[goal.quadrant])
                    if owner in (1, 2):
                        result[owner - 1] += self.settings["yellow_face_points"]
        return result

    def done(self) -> bool:
        return self.time >= self.settings["episode_seconds"] - EPS

    def clone(self) -> FieldState:
        return deepcopy(self)
