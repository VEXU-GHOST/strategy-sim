"""Event-driven high-level strategy simulator for VEX U Override.

This is a *surrogate* strategy environment, not an exact implementation of
physical nesting, field collisions, motion planning or official match loads.
It follows official VEX U period lengths and scoring magnitudes, and exposes
an explicit contract for swapping in a teammate's simulator later.
"""
from __future__ import annotations
from math import hypot
from typing import Callable
import numpy as np

from .actions import Action, ActionCodec, KINDS
from .config import load_config, total_seconds
from .types import Goal, Pending, Robot, Source, StackItem


class StrategyEnv:
    def __init__(self, config: dict | None = None, opponent: str = "none",
                 seed: int = 0, opponent_predictor: Callable | None = None):
        self.cfg = config if config is not None else load_config()
        if opponent not in {"none", "greedy", "coordination", "random", "mixed", "selfplay"}:
            raise ValueError(f"unsupported opponent: {opponent}")
        if opponent == "selfplay" and opponent_predictor is None:
            raise ValueError("selfplay requires a frozen opponent policy")
        self.codec = ActionCodec(self.cfg)
        self.opponent = opponent
        self.opponent_predictor = opponent_predictor
        self.rng = np.random.default_rng(seed)
        self.total_time = total_seconds(self.cfg)
        self.auto_time = float(self.cfg["autonomous_seconds"])
        from .features import StateEncoder
        self.encoder = StateEncoder(self.cfg)
        self.reset(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.done = False
        self.auto_done = False
        self.auto_bonus = [0.0, 0.0]
        self.awp_met = [False, False]
        self.awp_snapshot = [0.0, 0.0]
        self.robots: dict[str, Robot] = {}
        for name, pos in self.cfg["initial_robots"].items():
            dx, dy = self.rng.uniform(-0.12, 0.12, size=2)
            team = 0 if name.startswith("A") else 1
            color = "red_pin" if team == 0 else "blue_pin"
            self.robots[name] = Robot(name, team, (float(pos[0] + dx), float(pos[1] + dy)), pin=color)
        self.goals = {
            g["id"]: Goal(g["id"], tuple(map(float, g["position"])),
                          int(g["quadrant"]), g["owner"], int(g["max_stack_items"]))
            for g in self.cfg["goals"]
        }
        # Official field starts with Yellow Pins in neutral short Goals;
        # physical orientations are simplified to one visible yellow half.
        for goal in self.goals.values():
            if goal.owner == "neutral" and goal.quadrant >= 0:
                goal.stack.append(StackItem("yellow_pin", -1))
        self.toggles = {t["id"]: -1 for t in self.cfg["toggles"]}
        self.toggles_by_quadrant = {int(t["quadrant"]): t["id"] for t in self.cfg["toggles"]}
        self.toggle_pos = {t["id"]: tuple(map(float, t["position"])) for t in self.cfg["toggles"]}
        self.sources = {}
        for s in self.cfg["sources"]:
            self.sources[s["id"]] = Source(s["id"], tuple(map(float, s["position"])),
                                           {k: int(s[k]) for k in ("red_pin", "blue_pin", "yellow_pin", "cup")})
        self.stats = {
            "actions": [0, 0], "collected": [0, 0], "scored_pins": [0, 0],
            "scored_cups": [0, 0], "toggles_set": [0, 0], "failed_actions": [0, 0],
            "duplicate_targets": [0, 0], "wait_seconds": [0.0, 0.0],
            "trace": [],
        }
        self.active_robot = "A0"
        self._opponent_style = (self.rng.choice(["greedy", "coordination", "random"]).item()
                                if self.opponent == "mixed" else self.opponent)
        self._launch_opponents()
        return self.observation(), {"action_mask": self.action_mask(), "active_robot": self.active_robot}

    @staticmethod
    def distance(a, b) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

    def _team_color(self, team: int) -> str:
        return "red" if team == 0 else "blue"

    def _collect_kind(self, team: int, kind: str) -> str:
        return ("red_pin" if team == 0 else "blue_pin") if kind == "alliance_pin" else kind

    def _midfield(self, pos: tuple[float, float]) -> bool:
        half = 0.95  # surrogate footprint, not exact robot geometry
        return abs(pos[0] - 6.0) <= half and abs(pos[1] - 6.0) <= half

    def _midfield_counts(self) -> list[int]:
        return [sum(self._midfield(r.position) and r.team == team for r in self.robots.values())
                for team in (0, 1)]

    def _yellow_owner(self, goal: Goal, include_midfield: bool = False) -> int:
        if goal.quadrant >= 0:
            return self.toggles[self.toggles_by_quadrant[goal.quadrant]]
        if include_midfield:
            counts = self._midfield_counts()
            return (0 if counts[0] > counts[1] else 1 if counts[1] > counts[0] else -1)
        return -1

    def _goal_halves(self, goal: Goal, include_midfield: bool = False) -> list[int]:
        totals = [0, 0]
        for item in goal.stack:
            if item.kind == "red_pin":
                totals[0] += 1
            elif item.kind == "blue_pin":
                totals[1] += 1
            elif item.kind == "yellow_pin":
                owner = self._yellow_owner(goal, include_midfield)
                if owner in (0, 1):
                    totals[owner] += 1
        return totals

    def scores(self, include_midfield: bool = False) -> list[float]:
        colored = float(self.cfg["score_points"]["alliance_colored_half"])
        yellow = float(self.cfg["score_points"]["yellow_half"])
        scores = [float(x) for x in self.auto_bonus]
        for goal in self.goals.values():
            for item in goal.stack:
                if item.kind == "red_pin":
                    scores[0] += colored
                elif item.kind == "blue_pin":
                    scores[1] += colored
                elif item.kind == "yellow_pin":
                    owner = self._yellow_owner(goal, include_midfield)
                    if owner >= 0:
                        scores[owner] += yellow
        if include_midfield:
            middle_points = float(self.cfg["score_points"]["midfield_robot"])
            mid = self._midfield_counts()
            for team in (0, 1):
                scores[team] += middle_points * mid[team]
        return scores

    def _awp_state(self, team: int, include_midfield: bool = True) -> tuple[float, bool]:
        target_halves = int(self.cfg["awp"]["scored_halves"])
        target_goals = int(self.cfg["awp"]["goals_with_two_halves"])
        pin_halves = 0
        qualified_goals = 0
        for goal in self.goals.values():
            # Simplified autonomous line approximation: home x-half + midfield.
            on_home_side = goal.quadrant == -1 or (goal.position[0] <= 6 if team == 0 else goal.position[0] >= 6)
            if not on_home_side:
                continue
            count = self._goal_halves(goal, include_midfield)[team]
            pin_halves += count
            qualified_goals += int(count >= 2)
        mid = self._midfield_counts()[team]
        not_touching_wall = all(
            0.25 < self.robots[rid].position[0] < self.cfg["field_size"] - 0.25 and
            0.25 < self.robots[rid].position[1] < self.cfg["field_size"] - 0.25
            for rid in (("A0", "A1") if team == 0 else ("B0", "B1"))
        )
        p1 = min(1.0, pin_halves / max(1, target_halves))
        p2 = min(1.0, qualified_goals / max(1, target_goals))
        p3 = float(mid >= int(self.cfg["awp"]["required_midfield_robots"]))
        p4 = float(not_touching_wall)
        success = p1 >= 1 and p2 >= 1 and p3 == 1 and p4 == 1
        return float((p1 + p2 + p3 + p4) / 4), bool(success)

    def _end_autonomous(self) -> float:
        if self.auto_done:
            return 0.0
        # VEX U VUG5 includes robots in midfield when scoring the autonomous bonus.
        provisional = self.scores(include_midfield=True)
        bonus = float(self.cfg["score_points"]["autonomous_bonus"])
        if provisional[0] > provisional[1]:
            self.auto_bonus[0] += bonus
        elif provisional[1] > provisional[0]:
            self.auto_bonus[1] += bonus
        else:
            self.auto_bonus[0] += bonus / 2
            self.auto_bonus[1] += bonus / 2
        extra_reward = 0.0
        for team in (0, 1):
            progress, met = self._awp_state(team)
            self.awp_met[team] = met
            self.awp_snapshot[team] = progress
            if met:
                extra_reward += (1 if team == 0 else -1) * float(self.cfg["reward"]["awp_bonus"])
        self.auto_done = True
        return extra_reward

    def state(self, team: int = 0, active_robot: str | None = None) -> dict:
        our_ids = ("A0", "A1") if team == 0 else ("B0", "B1")
        opponent_ids = ("B0", "B1") if team == 0 else ("A0", "A1")
        now_scores = self.scores(include_midfield=self.done)
        robot_states = {}
        for rid, robot in self.robots.items():
            p = robot.pending
            robot_states[rid] = {
                "position": robot.position,
                "carried_pin_type": (None if robot.pin is None else
                                     "alliance_pin" if robot.pin == self._collect_kind(robot.team, "alliance_pin") else "yellow_pin"),
                "has_cup": robot.cup,
                "current_target": p.action.target if p else None,
                "current_action": p.action.verb if p else None,
                "target_position": p.destination if p else robot.position,
                "remaining_action_duration": max(0.0, p.finish_at - self.time) if p else 0.0,
                "busy": p is not None,
            }
        goals = []
        for g in self.goals.values():
            counted = self._goal_halves(g, include_midfield=self.done)
            goals.append({
                "goal_id": g.goal_id, "position": g.position, "quadrant_id": g.quadrant,
                "owner": g.owner, "our_scored_pins": counted[team],
                "opponent_scored_pins": counted[1-team],
                "yellow_pins": sum(item.kind == "yellow_pin" for item in g.stack),
                "our_alliance_pins": sum(item.kind == self._collect_kind(team, "alliance_pin") for item in g.stack),
                "opponent_alliance_pins": sum(item.kind == self._collect_kind(1-team, "alliance_pin") for item in g.stack),
                "cups": sum(item.kind == "cup" for item in g.stack),
                "stack_capacity": g.max_items, "remaining_capacity": g.max_items - len(g.stack),
                "can_place": g.can_add("red_pin"), "can_place_pin": g.can_add("red_pin"),
                "can_place_cup": g.can_add("cup"),
            })
        sources = []
        for source in self.sources.values():
            sources.append({
                "source_id": source.source_id, "position": source.position,
                "available_alliance_pins": source.available[self._collect_kind(team, "alliance_pin")],
                "available_opponent_pins": source.available[self._collect_kind(1-team, "alliance_pin")],
                "available_yellow_pins": source.available["yellow_pin"],
                "available_cups": source.available["cup"],
                "available": dict(source.available),
            })
        toggle_states = [
            {"toggle_id": t["id"], "quadrant_id": t["quadrant"],
             "position": tuple(t["position"]),
             "owner": "neutral" if self.toggles[t["id"]] == -1 else
                       "ours" if self.toggles[t["id"]] == team else "opponent"}
            for t in self.cfg["toggles"]
        ]
        awp_progress = self.awp_snapshot[team] if self.auto_done else self._awp_state(team)[0]
        elapsed_driver = max(0.0, self.time - self.auto_time)
        return {
            "time": self.time,
            "time_remaining": max(0.0, self.total_time - self.time),
            "phase": "autonomous" if not self.auto_done else
                     "endgame" if elapsed_driver >= self.cfg["driver_seconds"] - self.cfg["endgame_seconds"] else "driver",
            "score_difference": now_scores[team] - now_scores[1-team],
            "our_score": now_scores[team], "opponent_score": now_scores[1-team],
            "awp_progress": awp_progress, "awp_met": self.awp_met[team],
            "our_robot_ids": our_ids, "opponent_robot_ids": opponent_ids,
            "active_robot": active_robot,
            "robots": robot_states, "goals": goals, "toggles": toggle_states, "sources": sources,
        }

    def observation(self, team: int = 0, active_robot: str | None = None) -> np.ndarray:
        selected = active_robot if active_robot is not None else self.active_robot
        return self.encoder.encode(self.state(team, selected), team)

    def _has_reserved_goal(self, goal_id: str) -> bool:
        return any(r.pending is not None and r.pending.action.verb == "SCORE" and
                   r.pending.action.target == goal_id for r in self.robots.values())

    def action_mask(self, robot_id: str | None = None) -> np.ndarray:
        rid = robot_id or self.active_robot
        if rid is None:
            return np.zeros(len(self.codec), dtype=np.bool_)
        robot = self.robots[rid]
        result = np.zeros(len(self.codec), dtype=np.bool_)
        if robot.pending is not None or self.done:
            return result
        team = robot.team
        for i, act in enumerate(self.codec.actions):
            if act.verb == "COLLECT":
                stock = self.sources[act.target].available[self._collect_kind(team, act.object_type)]
                can_hold = (not robot.cup) if act.object_type == "cup" else robot.pin is None
                result[i] = stock > 0 and can_hold
            elif act.verb == "SCORE":
                goal = self.goals[act.target]
                permitted_goal = goal.owner in ("neutral", self._team_color(team))
                carried = robot.cup if act.object_type == "cup" else robot.pin == self._collect_kind(team, act.object_type)
                stack_kind = self._collect_kind(team, act.object_type)
                result[i] = bool(permitted_goal and carried and goal.can_add(stack_kind)
                                 and not self._has_reserved_goal(goal.goal_id))
            elif act.verb == "SET_TOGGLE":
                result[i] = self.toggles[act.target] != team
            elif act.verb == "GO_MIDFIELD":
                result[i] = not self._midfield(robot.position)
            elif act.verb == "WAIT":
                result[i] = True
            elif act.verb == "BLOCK":
                result[i] = False  # listed for the future, deliberately masked
        assert result.any(), f"WAIT must always remain valid for idle robot {rid}"
        return result

    def action_duration(self, robot_id: str, action: Action) -> float:
        if action.verb == "WAIT":
            return float(self.cfg["action_seconds"]["wait"])
        destination = self.destination(robot_id, action)
        travel = self.distance(self.robots[robot_id].position, destination) / float(self.cfg["robot_speed_units_per_second"])
        key = action.verb.lower()
        return travel + float(self.cfg["action_seconds"][key])

    def destination(self, robot_id: str, action: Action) -> tuple[float, float]:
        if action.verb == "COLLECT":
            return self.sources[action.target].position
        if action.verb == "SCORE":
            return self.goals[action.target].position
        if action.verb == "SET_TOGGLE":
            return self.toggle_pos[action.target]
        if action.verb == "GO_MIDFIELD":
            return (6.0, 6.0)
        if action.verb == "WAIT":
            return self.robots[robot_id].position
        raise NotImplementedError("BLOCK intentionally excluded from V1 strategy action mask")

    def _dispatch(self, robot_id: str, action_id: int):
        mask = self.action_mask(robot_id)
        if not 0 <= int(action_id) < len(self.codec) or not mask[int(action_id)]:
            raise ValueError(f"illegal action {action_id} for {robot_id} at t={self.time:.3f}")
        robot = self.robots[robot_id]
        action = self.codec.decode(action_id)
        teammate_target = any(
            teammate.team == robot.team and teammate.name != robot_id and
            teammate.pending is not None and teammate.pending.action.target and
            teammate.pending.action.target == action.target and
            teammate.pending.action.verb == action.verb
            for teammate in self.robots.values()
        ) if action.target else False
        if teammate_target:
            self.stats["duplicate_targets"][robot.team] += 1
        if action.verb == "COLLECT":
            # Reserve at dispatch, preventing two robots collecting last available item.
            key = self._collect_kind(robot.team, action.object_type)
            self.sources[action.target].available[key] -= 1
        duration = self.action_duration(robot_id, action)
        destination = self.destination(robot_id, action)
        robot.pending = Pending(action, self.time + duration, destination)
        self.stats["actions"][robot.team] += 1
        self.stats["trace"].append((round(self.time, 3), robot_id, str(action), round(duration, 3)))
        return bool(teammate_target)

    def _finish(self, robot: Robot):
        task = robot.pending
        assert task is not None
        act = task.action
        robot.position = task.destination
        robot.pending = None
        team = robot.team
        if act.verb == "COLLECT":
            if act.object_type == "cup":
                robot.cup = True
            else:
                robot.pin = self._collect_kind(team, act.object_type)
            self.stats["collected"][team] += 1
        elif act.verb == "SCORE":
            goal = self.goals[act.target]
            kind = self._collect_kind(team, act.object_type)
            if goal.can_add(kind):
                goal.stack.append(StackItem(kind, team))
                if kind == "cup":
                    robot.cup = False
                    self.stats["scored_cups"][team] += 1
                else:
                    robot.pin = None
                    self.stats["scored_pins"][team] += 1
            else:
                self.stats["failed_actions"][team] += 1  # race with simultaneous completion
        elif act.verb == "SET_TOGGLE":
            self.toggles[act.target] = team
            self.stats["toggles_set"][team] += 1
        elif act.verb == "WAIT":
            self.stats["wait_seconds"][team] += float(self.cfg["action_seconds"]["wait"])

    def _launch_opponents(self):
        if self.opponent == "none" or self.done:
            return
        from .policies import choose_action
        for rid in ("B0", "B1"):
            if self.robots[rid].pending is not None:
                continue
            mask = self.action_mask(rid)
            if self._opponent_style == "selfplay":
                obs = self.observation(team=1, active_robot=rid)
                canonical_mask = self.codec.canonical_mask(mask, team=1)
                canonical_choice = int(self.opponent_predictor(obs, canonical_mask))
                choice = self.codec.physical_index(canonical_choice, team=1)
                if not mask[choice]:
                    raise ValueError("selfplay model proposed an invalid action")
            else:
                choice = choose_action(self, rid, self._opponent_style, self.rng)
            self._dispatch(rid, choice)

    def _next_friendly_idle(self) -> str | None:
        for rid in ("A0", "A1"):
            if self.robots[rid].pending is None:
                return rid
        return None

    def _stats_info(self, dt: float) -> dict:
        scores = self.scores(include_midfield=self.done)
        return {
            "delta_time": dt,
            "game_time": self.time,
            "active_robot": self.active_robot,
            "our_score": scores[0], "opponent_score": scores[1],
            "score_difference": scores[0] - scores[1],
            "win": (1.0 if scores[0] > scores[1] else 0.0 if scores[0] < scores[1] else 0.5) if self.done else None,
            "awp_ours": self.awp_met[0], "awp_opponent": self.awp_met[1],
            "awp_progress": self.awp_snapshot[0] if self.auto_done else self._awp_state(0)[0],
            "duplicate_targets": self.stats["duplicate_targets"][0],
            "wait_seconds": self.stats["wait_seconds"][0],
            "scored_pins": self.stats["scored_pins"][0],
            "scored_cups": self.stats["scored_cups"][0],
            "collected": self.stats["collected"][0],
            "toggles_set": self.stats["toggles_set"][0],
            "failed_actions": self.stats["failed_actions"][0],
        }

    def step(self, action_id: int):
        if self.done or self.active_robot is None:
            raise RuntimeError("episode finished; call reset()")
        if self.active_robot not in {"A0", "A1"}:
            raise RuntimeError("the central policy controls only the two friendly robots")
        robot_id = self.active_robot
        duplicate = self._dispatch(robot_id, action_id)
        action = self.codec.decode(action_id)
        penalty = (float(self.cfg["reward"]["duplicate_target_penalty"]) if duplicate else 0.0)
        if action.verb == "WAIT":
            penalty += float(self.cfg["reward"]["wait_penalty_per_second"]) * float(self.cfg["action_seconds"]["wait"])
        next_idle = self._next_friendly_idle()
        if next_idle is not None:
            self.active_robot = next_idle
            return self.observation(), -penalty, False, False, self._stats_info(0.0)

        t_start = self.time
        old = self.scores()
        old_diff = old[0] - old[1]
        extra_reward = 0.0
        for _ in range(10000):  # hard stop on accidental non-progress
            times = [r.pending.finish_at for r in self.robots.values() if r.pending]
            if not self.auto_done:
                times.append(self.auto_time)
            times.append(self.total_time)
            event_time = min(times)
            if event_time <= self.time + 1e-10:
                raise RuntimeError("non-progressing event queue; check positive durations")
            self.time = min(event_time, self.total_time)
            ready = [r for r in self.robots.values() if r.pending and r.pending.finish_at <= self.time + 1e-9]
            for robot in sorted(ready, key=lambda r: r.name):
                self._finish(robot)
            if not self.auto_done and self.time >= self.auto_time - 1e-9:
                extra_reward += self._end_autonomous()
            if self.time >= self.total_time - 1e-9:
                self.done = True
                self.active_robot = None
                break
            self._launch_opponents()
            next_idle = self._next_friendly_idle()
            if next_idle is not None:
                self.active_robot = next_idle
                break
        else:
            raise RuntimeError("too many events in one decision step")

        new = self.scores(include_midfield=self.done)
        new_diff = new[0] - new[1]
        reward = ((new_diff - old_diff) * float(self.cfg["reward"]["score_difference_scale"])
                  + extra_reward - penalty)
        if self.done:
            reward += float(self.cfg["reward"]["terminal_win_bonus"]) * (1 if new_diff > 0 else -1 if new_diff < 0 else 0)
            # If a WAIT was interrupted by the clock, the logged time remains a lower-bound estimate.
        return (self.observation() if not self.done else self.encoder.empty(), float(reward),
                self.done, False, self._stats_info(self.time - t_start))
