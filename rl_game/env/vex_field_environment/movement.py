"""Allowance map, Dijkstra routing, and constant-speed pose samples.

Input: current (x,y,theta), target position/pose, static goals and walls.
Output: a MotionPlan containing a pose per tick, or ValueError on failure.
Robot/robot interference is resolved for all proposed poses together each tick.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .states import EPS, FieldState, Pose


def angle_difference(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def segment_distance(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length))
    return math.hypot(point[0] - start[0] - t * dx, point[1] - start[1] - t * dy)


class AllowanceMap:
    """A robot-center map with walls/goals inflated by the robot's radius."""

    def __init__(self, field: FieldState):
        self.field = field
        self.radius = float(field.settings["robot_radius"])
        self.goal_radius = float(field.settings["goal_radius"])
        self.goal_centers = [field.cell_to_world(g.row, g.col) for g in field.goals.values()]
        rows, cols = np.indices((field.board_size, field.board_size))
        x = cols / field.cells_per_foot - field.half_width
        y = field.half_height - rows / field.cells_per_foot
        self.allowed = (np.abs(x) <= field.half_width - self.radius + EPS) & (np.abs(y) <= field.half_height - self.radius + EPS)
        for gx, gy in self.goal_centers:
            self.allowed &= (x - gx) ** 2 + (y - gy) ** 2 >= (self.radius + self.goal_radius - EPS) ** 2

    def point_clear(self, point) -> bool:
        x, y = point
        return (math.isfinite(x) and math.isfinite(y)
                and abs(x) <= self.field.half_width - self.radius + EPS
                and abs(y) <= self.field.half_height - self.radius + EPS
                and all(math.dist(point, center) >= self.radius + self.goal_radius - EPS for center in self.goal_centers))

    def segment_clear(self, start, end) -> bool:
        return (self.point_clear(start) and self.point_clear(end)
                and all(segment_distance(center, start, end) >= self.radius + self.goal_radius - EPS
                        for center in self.goal_centers))

    def _nearby_cells(self, point):
        row, col = self.field.world_to_cell(point)
        for r in range(max(0, row - 1), min(self.field.board_size, row + 2)):
            for c in range(max(0, col - 1), min(self.field.board_size, col + 2)):
                xy = self.field.cell_to_world(r, c)
                if self.allowed[r, c] and self.segment_clear(point, xy):
                    yield (r, c), math.dist(point, xy)

    def _search(self, start, end_cost, exact_end=None):
        if not self.point_clear(start):
            raise ValueError("start_position_blocked")
        costs, parents, heap = {}, {}, []
        for cell, distance in self._nearby_cells(start):
            costs[cell], parents[cell] = distance, None
            heapq.heappush(heap, (distance, cell))
        best, finish = math.inf, None
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
        while heap:
            distance, cell = heapq.heappop(heap)
            if distance > costs[cell] + EPS:
                continue
            if distance > best + EPS:
                break
            xy = self.field.cell_to_world(*cell)
            tail = end_cost(cell, xy)
            if tail is not None and distance + tail < best:
                best, finish = distance + tail, cell
            for dr, dc in neighbors:
                other = cell[0] + dr, cell[1] + dc
                if not (0 <= other[0] < self.field.board_size and 0 <= other[1] < self.field.board_size):
                    continue
                if not self.allowed[other]:
                    continue
                other_xy = self.field.cell_to_world(*other)
                candidate = distance + math.hypot(dr, dc) / self.field.cells_per_foot
                if candidate + EPS < costs.get(other, math.inf) and self.segment_clear(xy, other_xy):
                    costs[other], parents[other] = candidate, cell
                    heapq.heappush(heap, (candidate, other))
        if finish is None:
            raise ValueError("target_unreachable")
        route = []
        while finish is not None:
            route.append(self.field.cell_to_world(*finish))
            finish = parents[finish]
        route = [tuple(start)] + list(reversed(route))
        if exact_end is not None:
            route.append(tuple(exact_end))
        # Remove redundant grid turns, checking the entire shortcut each time.
        result = [route[0]]
        index = 0
        while index < len(route) - 1:
            following = len(route) - 1
            while following > index + 1 and not self.segment_clear(route[index], route[following]):
                following -= 1
            if math.dist(result[-1], route[following]) > EPS:
                result.append(route[following])
            index = following
        return result

    def route(self, start, destination):
        destination = tuple(map(float, destination))
        if not self.point_clear(destination):
            raise ValueError("target_position_blocked")
        if self.segment_clear(start, destination):
            return [tuple(start)] if math.dist(start, destination) < EPS else [tuple(start), destination]
        ends = dict(self._nearby_cells(destination))
        return self._search(start, lambda cell, xy: ends.get(cell), destination)

    def approach(self, start, target, reach: float):
        """Reach an interaction position, without driving into a goal/wall."""
        distance = math.dist(start, target)
        if distance <= reach + EPS and self.point_clear(start):
            return [tuple(start)]
        desired_distance = reach * 0.95
        approach = tuple(target[i] + (start[i] - target[i]) * desired_distance / distance for i in range(2))
        if self.segment_clear(start, approach):
            return [tuple(start), approach]
        return self._search(start, lambda cell, xy: 0.0 if math.dist(xy, target) <= reach + EPS else None)

    def park_route(self, start, bounds):
        xmin, xmax, ymin, ymax = bounds
        xmin, xmax = xmin + self.radius, xmax - self.radius
        ymin, ymax = ymin + self.radius, ymax - self.radius
        target = (min(xmax, max(xmin, start[0])), min(ymax, max(ymin, start[1])))
        if self.segment_clear(start, target):
            return [tuple(start)] if math.dist(start, target) < EPS else [tuple(start), target]
        return self._search(start, lambda cell, xy: 0.0 if xmin - EPS <= xy[0] <= xmax + EPS and ymin - EPS <= xy[1] <= ymax + EPS else None)


@dataclass(frozen=True)
class MotionPlan:
    poses: tuple[Pose, ...]
    tick_seconds: float

    @property
    def duration(self) -> float:
        return max(0, len(self.poses) - 1) * self.tick_seconds


def sample_motion(start: Pose, route, speed: float, turn_speed: float, tick: float,
                  final_heading: float | None = None) -> MotionPlan:
    """Turn in place, drive each clear segment, then optionally align heading.

    Both rates are constant up to the final partial increment of each segment.
    Poses include the initial pose. Dijkstra minimizes travel distance; it does
    not optimize a robot's turn time or model its drivetrain dynamics.
    """
    samples = [start]

    def turn_to(target):
        if not math.isfinite(target):
            raise ValueError("heading_must_be_finite")
        delta = angle_difference(target, samples[-1].theta)
        while abs(delta) > EPS:
            increment = math.copysign(min(abs(delta), turn_speed * tick), delta)
            old = samples[-1]
            samples.append(Pose(old.x, old.y, (old.theta + increment) % 360))
            delta = angle_difference(target, samples[-1].theta)

    for target in route[1:]:
        old = samples[-1]
        distance = math.dist(old.xy, target)
        if distance <= EPS:
            continue
        direction = ((target[0] - old.x) / distance, (target[1] - old.y) / distance)
        turn_to(math.degrees(math.atan2(direction[1], direction[0])))
        origin, traveled = samples[-1], 0.0
        while traveled < distance - EPS:
            traveled = min(distance, traveled + speed * tick)
            samples.append(Pose(origin.x + direction[0] * traveled,
                                origin.y + direction[1] * traveled, origin.theta))
    if final_heading is not None:
        turn_to(float(final_heading))
    return MotionPlan(tuple(samples), tick)


def interpolate(start: Pose, end: Pose, fraction: float) -> Pose:
    return Pose(start.x + (end.x - start.x) * fraction,
                start.y + (end.y - start.y) * fraction,
                (start.theta + angle_difference(end.theta, start.theta) * fraction) % 360)


def collision_stops(field: FieldState, allowance: AllowanceMap, proposals: dict[str, Pose]) -> set[str]:
    """Resolve all robots from the same snapshot, including crossing paths."""
    starts = {rid: field.pose(rid) for rid in field.robots}
    ends = {rid: proposals.get(rid, pose) for rid, pose in starts.items()}
    stopped = {rid for rid in proposals if not allowance.segment_clear(starts[rid].xy, ends[rid].xy)}
    for rid in stopped:
        ends[rid] = starts[rid]
    ids, changed = sorted(field.robots), True
    while changed:
        newly_stopped = set()
        for index, left in enumerate(ids):
            for right in ids[index + 1:]:
                a = (starts[left].x - starts[right].x, starts[left].y - starts[right].y)
                b = (ends[left].x - ends[right].x, ends[left].y - ends[right].y)
                if segment_distance((0, 0), a, b) < 2 * allowance.radius - EPS:
                    newly_stopped.update(rid for rid in (left, right) if ends[rid].xy != starts[rid].xy)
        newly_stopped -= stopped
        changed = bool(newly_stopped)
        stopped.update(newly_stopped)
        for rid in newly_stopped:
            ends[rid] = starts[rid]
    return stopped
