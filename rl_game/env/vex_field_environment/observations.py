"""Read-only observations: V0/full field or V1/range-and-cone sensors.

These V0/V1 names refer to the whiteboard, not Amy's v1 action-wrapper API.
Cone observations omit the full board and out-of-view dynamic objects. Time
and the public scoreboard remain available. They include goal-base occlusion,
but not camera noise, robot occlusion, lighting or pin-half image recognition.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .actions import CLAW_ACTIONS
from .movement import angle_difference, segment_distance
from .states import EPS, ITEM_TYPES, FieldState


def robot_state(field: FieldState, rid: str) -> dict:
    robot, pose = field.robots[rid], field.pose(rid)
    task = robot.task
    return {"team": ("red", "blue")[robot.team], "position": pose.xy,
            "row": robot.row, "col": robot.col, "heading": pose.theta,
            "held_items": robot.held_items.copy(), "parked": robot.parked,
            "busy": task is not None,
            "phase": ("moving" if task.moving else "interacting") if task else "idle",
            "current_action": task.action.verb if task else None,
            "target": task.action.target if task else None,
            "remaining_time": task.remaining_seconds(field.settings["seconds_per_step"]) if task else 0.0,
            "wheels_active": bool(task and task.moving),
            "claw_active": bool(task and not task.moving and task.action.verb in CLAW_ACTIONS)}


def goal_state(field: FieldState, goal) -> dict:
    return {"position": field.cell_to_world(goal.row, goal.col), "quadrant": goal.quadrant,
            "capacity": goal.max_items, "stack": [
                {"item": ITEM_TYPES[ch], "face": face} for ch, face in goal.stack]}


def toggle_state(field: FieldState, toggle: dict) -> dict:
    return {**deepcopy(toggle), "owner": int(field.zone_colors[toggle["quadrant"]])}


def loader_state(loader) -> dict:
    return {"position": loader.position,
            "team": ("red", "blue")[loader.team] if loader.team is not None else None,
            "stock": {name: int(loader.stock[i]) for i, name in enumerate(ITEM_TYPES)}}


def full_state(field: FieldState) -> dict:
    """Independent snapshot suitable for debugging or fully observed planning."""
    return {"mode": "full", "time": field.time,
            "time_remaining": max(0.0, field.settings["episode_seconds"] - field.time),
            "board": field.board.copy(), "objects": field.object_states(),
            "robots": {rid: robot_state(field, rid) for rid in field.robots},
            "goals": {gid: goal_state(field, goal) for gid, goal in field.goals.items()},
            "goalStates": field.goal_states(), "zoneColors": field.zone_colors.copy(),
            "toggles": {tid: toggle_state(field, toggle) for tid, toggle in field.toggles.items()},
            "matchloaders": {lid: loader_state(loader) for lid, loader in field.loaders.items()},
            "parking_zones": deepcopy(field.parking_zones),
            "loaderStates": [field.loader_stock(0), field.loader_stock(1)],
            "scores": field.scores(), "done": field.done()}


def cone_state(field: FieldState, observer_id: str) -> dict:
    observer = field.pose(observer_id)
    reach, fov = field.settings["sensor_range"], field.settings["sensor_fov_degrees"]

    def visible(point, ignore_goal=None):
        distance = math.dist(observer.xy, point)
        if distance <= EPS:
            return True
        bearing = math.degrees(math.atan2(point[1] - observer.y, point[0] - observer.x))
        if distance > reach + EPS or abs(angle_difference(bearing, observer.theta)) > fov / 2 + EPS:
            return False
        for gid, goal in field.goals.items():
            if gid == ignore_goal:
                continue
            center = field.cell_to_world(goal.row, goal.col)
            if segment_distance(center, observer.xy, point) < field.settings["goal_radius"] - EPS:
                return False
        return True

    other_robots = {}
    for rid, robot in field.robots.items():
        if rid != observer_id and visible(field.pose(rid).xy):
            pose = field.pose(rid)
            # A visible opponent does not expose its pending task or inventory.
            other_robots[rid] = {"position": pose.xy, "heading": pose.theta,
                                 "team": ("red", "blue")[robot.team]}
    return {"mode": "cone", "observer_id": observer_id, "time": field.time,
            "time_remaining": max(0.0, field.settings["episode_seconds"] - field.time),
            "scores": field.scores(), "done": field.done(),
            "self": robot_state(field, observer_id), "robots": other_robots,
            "sensor_range": reach, "sensor_fov_degrees": fov,
            "objects": [obj for obj in field.object_states() if visible(obj["position"])],
            "goals": {gid: goal_state(field, goal) for gid, goal in field.goals.items()
                      if visible(field.cell_to_world(goal.row, goal.col), gid)},
            "toggles": {tid: toggle_state(field, toggle) for tid, toggle in field.toggles.items()
                        if visible(toggle["position"])},
            "matchloaders": {lid: loader_state(loader) for lid, loader in field.loaders.items()
                             if visible(loader.position)}}


def observe(field: FieldState, robot_id: str | None = None, mode="full") -> dict:
    if mode in ("full", "v0", 0):
        return full_state(field)
    if mode in ("cone", "v1", 1):
        if robot_id not in field.robots:
            raise ValueError("A cone observation requires a valid robot_id")
        return cone_state(field, robot_id)
    raise ValueError("Observation mode must be full/v0 or cone/v1")
