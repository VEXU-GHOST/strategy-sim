"""Tasks: drive to a target, operate the claw, then change the world.

Preparation and duration estimates are read-only. Effects happen only when the
coordinator completes a task. Stock, cargo, reach and capacity are rechecked at
completion, since other robots may have changed them during the trip.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .movement import AllowanceMap, sample_motion
from .states import COLORS, EPS, ITEM_TYPES, PIN_FACES, FieldState, PendingTask

VERBS = {"MOVE", "TURN", "COLLECT", "DROP", "SCORE", "TOGGLE", "MATCHLOAD", "PARK", "WAIT"}
CLAW_ACTIONS = {"COLLECT", "DROP", "SCORE", "TOGGLE", "MATCHLOAD"}


@dataclass(frozen=True)
class Action:
    verb: str
    target: str | None = None
    position: tuple[float, float] | None = None
    item: str | None = None
    face: str | None = None
    angle: float = 0.0              # TURN increment, -180..180 degrees.
    heading: float | None = None     # Optional final heading for a task.

    def __post_init__(self):
        verb = self.verb.upper()
        verb = {"PICKUP": "COLLECT", "IDLE": "WAIT"}.get(verb, verb)
        if verb == "BLOCK":
            raise ValueError("BLOCK is reserved for a later version")
        if verb not in VERBS:
            raise ValueError(f"Unknown action: {verb}")
        object.__setattr__(self, "verb", verb)


def _channel(field: FieldState, rid: str, action: Action) -> int:
    held = field.robots[rid].held_items
    if action.item is not None:
        if action.item not in ITEM_TYPES:
            raise ValueError("unknown_item_type")
        channel = ITEM_TYPES.index(action.item)
    else:
        channels = np.flatnonzero(held > 0)
        if len(channels) != 1:
            raise ValueError("specify_item_when_cargo_is_empty_or_ambiguous")
        channel = int(channels[0])
    if held[channel] <= 0:
        raise ValueError("robot_not_carrying_that_item")
    return channel


def _check_slot(field: FieldState, rid: str, channel: int) -> None:
    held = field.robots[rid].held_items
    if (channel == 4 and held[4] > 0) or (channel < 4 and held[:4].sum() > 0):
        raise ValueError("cargo_slot_full")


def _check_score(field: FieldState, rid: str, action: Action):
    channel = _channel(field, rid, action)
    goal = field.goals.get(action.target)
    if goal is None:
        raise ValueError("unknown_goal")
    if len(goal.stack) >= goal.max_items:
        raise ValueError("goal_full")
    if field.settings["alternate_pin_and_cup"]:
        needs_cup = bool(goal.stack and goal.stack[-1][0] != 4)
        if (channel == 4) != needs_cup:
            raise ValueError("goal_requires_cup" if needs_cup else "goal_requires_pin")
    face = None
    if channel < 4:
        our_color = ("red", "blue")[field.robots[rid].team]
        face = action.face or (our_color if our_color in PIN_FACES[channel] else "yellow")
        if face not in PIN_FACES[channel]:
            raise ValueError("that_face_is_not_on_this_pin")
    return channel, goal, face


def _loader(field: FieldState, rid: str, action: Action):
    loader = field.loaders.get(action.target)
    if loader is None:
        raise ValueError("unknown_matchloader")
    if loader.team is not None and loader.team != field.robots[rid].team:
        raise ValueError("opponent_matchloader")
    if action.item not in ITEM_TYPES:
        raise ValueError("MATCHLOAD_requires_a_valid_item")
    channel = ITEM_TYPES.index(action.item)
    _check_slot(field, rid, channel)
    if loader.stock[channel] < 1:
        raise ValueError("matchloader_empty")
    return loader, channel


def _near(field: FieldState, rid: str, target) -> None:
    if math.dist(field.pose(rid).xy, target) > field.settings["interaction_reach"] + EPS:
        raise ValueError("target_out_of_reach")


def _parking_zone(field: FieldState, rid: str, target: str):
    zone = field.parking_zones.get(target)
    if zone is None:
        raise ValueError("parking_zone_not_configured")
    if zone.get("team") is not None and COLORS[zone["team"]] != field.robots[rid].team:
        raise ValueError("opponent_parking_zone")
    return zone


def prepare_action(field: FieldState, allowance: AllowanceMap, rid: str, action: Action) -> PendingTask:
    """Build a path and dwell time without modifying inventory or the clock."""
    if rid not in field.robots:
        raise ValueError("unknown_robot")
    if field.robots[rid].task is not None:
        raise ValueError("robot_busy")
    start = field.pose(rid)
    route, target = [start.xy], None
    dwell = 0.0
    heading = action.heading
    if action.verb == "MOVE":
        if action.position is None:
            raise ValueError("MOVE_requires_position")
        route = allowance.route(start.xy, action.position)
    elif action.verb == "TURN":
        if not math.isfinite(action.angle) or abs(action.angle) > 180:
            raise ValueError("TURN_angle_must_be_between_minus_180_and_180")
        heading = start.theta + action.angle
    elif action.verb == "COLLECT":
        obj = field.find_object(action.target)
        _check_slot(field, rid, obj["channel"])
        target, dwell = obj["position"], field.settings["collect_seconds"]
    elif action.verb == "SCORE":
        _, goal, _ = _check_score(field, rid, action)
        target = field.cell_to_world(goal.row, goal.col)
        dwell = field.settings["score_seconds"]
    elif action.verb == "TOGGLE":
        toggle = field.toggles.get(action.target)
        if toggle is None:
            raise ValueError("toggle_not_configured")
        if field.zone_colors[toggle["quadrant"]] == field.robots[rid].team + 1:
            raise ValueError("toggle_already_ours")
        target, dwell = toggle["position"], field.settings["toggle_seconds"]
    elif action.verb == "MATCHLOAD":
        loader, _ = _loader(field, rid, action)
        target, dwell = loader.position, field.settings["matchload_seconds"]
    elif action.verb == "DROP":
        _channel(field, rid, action)
        dwell = field.settings["drop_seconds"]
    elif action.verb == "PARK":
        zone = _parking_zone(field, rid, action.target)
        route = allowance.park_route(start.xy, zone["bounds"])
        dwell = field.settings["park_seconds"]
    else:
        dwell = field.settings["wait_seconds"]

    if target is not None:
        route = allowance.approach(start.xy, target, field.settings["interaction_reach"])
        if heading is None and math.dist(route[-1], target) > EPS:
            heading = math.degrees(math.atan2(target[1] - route[-1][1], target[0] - route[-1][0]))
    tick = field.settings["seconds_per_step"]
    plan = sample_motion(start, route, field.settings["robot_speed"], field.settings["turn_speed"], tick, heading)
    if plan.duration < EPS and dwell < EPS:
        dwell = tick  # A no-op MOVE/TURN still occupies one tick.
    dwell_ticks = math.ceil(max(0.0, dwell - EPS) / tick)
    return PendingTask(action, plan.poses, dwell, field.time, plan.duration + dwell_ticks * tick)


def estimate_action(field: FieldState, allowance: AllowanceMap, rid: str, action: Action) -> dict:
    """Action consequences: estimated arrival/finish times, without execution."""
    task = prepare_action(field, allowance, rid, action)
    travel = (len(task.poses) - 1) * field.settings["seconds_per_step"]
    return {"duration_seconds": task.estimated_duration, "movement_seconds": travel,
            "interaction_seconds": task.estimated_duration - travel,
            "finish_at": field.time + task.estimated_duration,
            "fits_before_match_end": field.time + task.estimated_duration <= field.settings["episode_seconds"] + EPS,
            "target_position": task.poses[-1].xy,
            "target_heading": task.poses[-1].theta}


def complete_action(field: FieldState, rid: str, action: Action) -> str:
    """Apply one completed interaction atomically after every robot has moved."""
    robot = field.robots[rid]
    if action.verb in {"MOVE", "TURN", "WAIT"}:
        return f"{action.verb.lower()}_complete"
    if action.verb == "COLLECT":
        obj = field.find_object(action.target)
        _near(field, rid, obj["position"])
        _check_slot(field, rid, obj["channel"])
        field.board[obj["channel"], obj["row"], obj["col"]] -= 1
        robot.held_items[obj["channel"]] += 1
    elif action.verb == "SCORE":
        channel, goal, face = _check_score(field, rid, action)
        _near(field, rid, field.cell_to_world(goal.row, goal.col))
        goal.stack.append((channel, face))
        robot.held_items[channel] -= 1
    elif action.verb == "DROP":
        channel = _channel(field, rid, action)
        row, col = field.world_to_cell(field.pose(rid).xy)
        field.board[channel, row, col] += 1
        robot.held_items[channel] -= 1
    elif action.verb == "TOGGLE":
        toggle = field.toggles[action.target]
        _near(field, rid, toggle["position"])
        field.zone_colors[toggle["quadrant"]] = robot.team + 1
    elif action.verb == "MATCHLOAD":
        loader, channel = _loader(field, rid, action)
        _near(field, rid, loader.position)
        loader.stock[channel] -= 1
        robot.held_items[channel] += 1
    elif action.verb == "PARK":
        zone = _parking_zone(field, rid, action.target)
        xmin, xmax, ymin, ymax = zone["bounds"]
        radius, position = field.settings["robot_radius"], field.pose(rid).xy
        if not (xmin + radius - EPS <= position[0] <= xmax - radius + EPS
                and ymin + radius - EPS <= position[1] <= ymax - radius + EPS):
            raise ValueError("robot_not_inside_parking_zone")
        robot.parked = True
    return f"{action.verb.lower()}_complete"
