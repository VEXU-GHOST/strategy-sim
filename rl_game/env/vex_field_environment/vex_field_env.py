"""Coordinator: dispatch tasks, move all robots, resolve effects, advance time.

The simulation clock lives here; states/actions/movement/observations are
separate modules corresponding to the whiteboard. No agent or trainer runs
when this package is imported.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .actions import Action, complete_action, estimate_action, prepare_action
from .movement import AllowanceMap, collision_stops, interpolate
from .observations import full_state, observe
from .states import EPS, FieldState


class GameEnv:
    def __init__(self, config_path: str | Path | None = None, *, config: dict | None = None):
        if config is not None and config_path is not None:
            raise ValueError("Supply config or config_path, not both")
        if config is None:
            path = Path(config_path) if config_path is not None else Path(__file__).with_name("field_config.json")
            with path.open(encoding="utf-8") as source:
                config = json.load(source)
        self.config = deepcopy(config)
        self.reset()

    def reset(self):
        self.field_state = FieldState(self.config)
        self.allowance = AllowanceMap(self.field_state)
        ids = list(self._agents)
        for i, rid in enumerate(ids):
            position = self.position(rid)
            if not self.allowance.point_clear(position):
                raise ValueError(f"{rid} spawns in a goal or wall clearance")
            if any(math.dist(position, self.position(other)) < 2 * self.settings["robot_radius"] - EPS
                   for other in ids[i + 1:]):
                raise ValueError(f"{rid} overlaps another robot at spawn")
        return self.state()

    def position(self, rid):
        return self.field_state.pose(rid).xy

    def world_to_cell(self, position):
        return self.field_state.world_to_cell(position)

    def cell_to_world(self, row, col):
        return self.field_state.cell_to_world(row, col)

    def object_states(self):
        return self.field_state.object_states()

    def state(self):
        return full_state(self.field_state)

    def observe(self, robot_id=None, mode="full"):
        return observe(self.field_state, robot_id, mode)

    def observe_all(self, mode="full"):
        return {rid: self.observe(rid, mode) for rid in self._agents}

    def estimate_action(self, rid, action):
        return estimate_action(self.field_state, self.allowance, rid, action)

    def start_action(self, rid, action):
        """Dispatch without advancing time, so allies can start together."""
        if self.done():
            raise RuntimeError("Match finished; reset before dispatching")
        if not isinstance(action, Action):
            raise ValueError("Expected an Action")
        task = prepare_action(self.field_state, self.allowance, rid, action)
        self._agents[rid].task = task
        return task

    def step(self, actions: dict[str, Action] | None = None):
        """Start idle robots' tasks, then advance ONE shared simulation tick.

        Omitted robots continue a pending task, or remain idle. The returned
        results distinguish 'running' from 'complete'/'failed'. Do not resubmit
        an action while its robot is busy. Multiple robots progress concurrently.
        """
        if self.done():
            raise RuntimeError("Match finished; call reset()")
        actions = {} if actions is None else actions
        for rid, action in actions.items():
            if rid not in self._agents or not isinstance(action, Action):
                raise ValueError(f"Unknown robot or invalid Action: {rid}")
        field = self.field_state
        before, results, events = field.scores(), {}, []
        # Prepare from the same field snapshot; preparation is read-only.
        prepared = {}
        for rid, action in actions.items():
            try:
                prepared[rid] = prepare_action(field, self.allowance, rid, action)
            except ValueError as exc:
                results[rid] = {"status": "rejected", "success": False, "reason": str(exc)}
        for rid, task in prepared.items():
            field.robots[rid].task = task

        tick = self.settings["seconds_per_step"]
        dt = min(tick, self.settings["episode_seconds"] - field.time)
        moving = {rid for rid, robot in field.robots.items() if robot.task and robot.task.moving}
        proposals = {}
        for rid in moving:
            task = field.robots[rid].task
            proposals[rid] = interpolate(field.pose(rid), task.poses[task.path_index + 1], dt / tick)
        stopped = collision_stops(field, self.allowance, proposals)

        # First update every robot pose. No interaction can see a half-updated world.
        for rid, pose in proposals.items():
            robot, task = field.robots[rid], field.robots[rid].task
            if rid in stopped:
                robot.task = None
                event = {"robot_id": rid, "status": "failed", "success": False, "reason": "motion_collision"}
                events.append(event)
                results[rid] = {key: value for key, value in event.items() if key != "robot_id"}
                continue
            if math.dist(field.pose(rid).xy, pose.xy) > EPS:
                robot.parked = False
            field.set_pose(rid, pose)
            if dt >= tick - EPS:
                task.path_index += 1

        # Only tasks already in their interaction phase at tick start can dwell.
        ready = []
        for rid, robot in field.robots.items():
            task = robot.task
            if task is None:
                continue
            if rid not in moving:
                task.interaction_remaining = max(0.0, task.interaction_remaining - dt)
            if not task.moving and task.interaction_remaining <= EPS:
                ready.append(rid)
        field.time = min(self.settings["episode_seconds"], field.time + dt)
        field.step_count += 1

        # Then commit completed interactions, rechecking stock/capacity. Equal
        # completion times use sorted robot ID, never caller dictionary order.
        for rid in sorted(ready):
            task = field.robots[rid].task
            try:
                reason = complete_action(field, rid, task.action)
                result = {"status": "complete", "success": True, "reason": reason}
            except ValueError as exc:
                result = {"status": "failed", "success": False, "reason": str(exc)}
            field.robots[rid].task = None
            results[rid] = result
            events.append({"robot_id": rid, "action": task.action.verb, **result})

        for rid, robot in field.robots.items():
            if robot.task is not None:
                if field.done():
                    robot.task = None
                    result = {"status": "failed", "success": False, "reason": "match_ended_before_completion"}
                    results[rid] = result
                    events.append({"robot_id": rid, **result})
                elif rid not in results:
                    results[rid] = {"status": "running", "success": None,
                                    "remaining_time": robot.task.remaining_seconds(tick)}
        after = field.scores()
        rewards = {rid: after[robot.team] - before[robot.team] for rid, robot in field.robots.items()}
        # Observations are built only after all poses, effects and time agree.
        return self.state(), rewards, field.done(), {"delta_time": dt, "results": results,
                                                   "events": events, "scores": after}

    def done(self):
        return self.field_state.done()

    def clone(self):
        return deepcopy(self)

    def calcGoalScore(self):
        return self.field_state.scores()

    # Raw-state compatibility for Amy's original V1 bridge. The newer PPO
    # feature/action interface still requires an adapter, as before.
    time = property(lambda self: self.field_state.time)
    settings = property(lambda self: self.field_state.settings)
    goals = property(lambda self: self.field_state.goals)
    toggles = property(lambda self: self.field_state.toggles)
    _board = property(lambda self: self.field_state.board)
    _board_size = property(lambda self: self.field_state.board_size)
    _agents = property(lambda self: self.field_state.robots)
    goalStates = property(lambda self: self.field_state.goal_states())
    goalLen = property(lambda self: np.array([len(g.stack) for g in self.goals.values()], dtype=int))
    zoneColors = property(lambda self: self.field_state.zone_colors)
    redMatchload = property(lambda self: self.field_state.loader_stock(0))
    blueMatchload = property(lambda self: self.field_state.loader_stock(1))
    fieldObject = property(lambda self: np.array([
        (g.row, g.col, self.settings["goal_radius"] * self.field_state.cells_per_foot)
        for g in self.goals.values()], dtype=float).reshape(-1, 3))
