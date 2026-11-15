"""StrategyEnv-compatible adapter over the tick-based field engine.

``vex_field_environment.GameEnv`` (rl_game/env, unmodified) owns geometry,
routing, motion, collisions, cargo, goal stacks, Toggles and Loaders.
This module adds the Override rules the engine leaves out (SG6, SG9, SG12,
SC5-SC7, preloads, driver-only Match Loads, AWP) and maps the 64 strategy
actions onto native engine actions.

Engine coordinates are centered feet; the strategy state uses the
surrogate's 0..12 frame (``+half``). Team 0 = red = A0/A1.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import heapq
import math
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from .actions import Action, ActionCodec
from .config import FIELD_CONFIG, load_config, total_seconds
from .features import StateEncoder
from .policies import choose_action
from .types import Goal, Pending, Robot, Source, StackItem

_ENV_DIR = Path(os.environ.get("VEX_FIELD_ENV_PATH", Path(__file__).resolve().parents[2] / "env"))
if str(_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_ENV_DIR))
from vex_field_environment import Action as NativeAction, GameEnv  # noqa: E402
from vex_field_environment.actions import prepare_action  # noqa: E402
from vex_field_environment.movement import AllowanceMap, segment_distance  # noqa: E402
from vex_field_environment.states import EPS, ITEM_TYPES, PIN_FACES  # noqa: E402

TEAM_COLOR = ("red", "blue")
CUP = ITEM_TYPES.index("cup")
ZONES = ("west", "north", "east", "south", "midfield")
FRIENDLY, OPPONENTS = ("A0", "A1"), ("B0", "B1")
OPPONENT_STYLES = {"none", "greedy", "coordination", "random", "mixed", "selfplay"}
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))  # engine order


def _point_in_polygon(point, polygon) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _shifted(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r, c] = a[r + dr, c + dc], False outside the board."""
    n, m = a.shape
    out = np.zeros_like(a)
    out[max(0, -dr):n - max(0, dr), max(0, -dc):m - max(0, dc)] = \
        a[max(0, dr):n + min(0, dr), max(0, dc):m + min(0, dc)]
    return out


def _edges_clear(x, y, ex, ey, obstacles) -> np.ndarray:
    """Vectorized engine ``segment_distance`` test for grid edges (x,y)->(x+ex,y+ey)."""
    clear = np.ones(x.shape, dtype=bool)
    length = ex * ex + ey * ey
    for ox, oy, c in obstacles:
        px, py = ox - x, oy - y
        t = np.clip((px * ex + py * ey) / length, 0.0, 1.0)
        clear &= np.hypot(px - t * ex, py - t * ey) >= c - EPS
    return clear


@lru_cache(maxsize=8)
def _static_edges(n: int, cpf: float, half: float, goals: tuple) -> tuple[np.ndarray, ...]:
    """Per-direction grid edges that clear every goal; computed once per layout."""
    rows, cols = np.indices((n, n))
    x, y = cols / cpf - half, half - rows / cpf
    return tuple(_edges_clear(x, y, dc / cpf, -dr / cpf, goals) for dr, dc in NEIGHBORS)


class RobotAwareAllowance(AllowanceMap):
    """The engine's allowance map plus the other robots' current poses.

    The engine plans around goals and walls only and fails a task on robot
    contact, so blocked robots would replan into each other every tick.
    ``_search`` is the engine's Dijkstra (same order and tie-breaks) over
    precomputed edge tables instead of per-edge geometry checks.
    """

    def __init__(self, field, obstacles):
        """obstacles: [((x, y), clearance)] between robot centers."""
        super().__init__(field)
        n, cpf, half = field.board_size, field.cells_per_foot, field.half_width
        goals = tuple((gx, gy, self.radius + self.goal_radius) for gx, gy in self.goal_centers)
        robots = [(float(ox), float(oy), float(c)) for (ox, oy), c in obstacles]
        self._obstacles = goals + tuple(robots)
        self._limit = half - self.radius + EPS

        edges = [e.copy() for e in _static_edges(n, cpf, half, goals)]
        reach = 0.25  # > longest grid edge (sqrt(2)/6 ft)
        for ox, oy, c in robots:
            r0, r1 = max(0, math.floor((half - oy - c - reach) * cpf)), min(n, math.ceil((half - oy + c + reach) * cpf) + 1)
            c0, c1 = max(0, math.floor((ox - c - reach + half) * cpf)), min(n, math.ceil((ox + c + reach + half) * cpf) + 1)
            rows, cols = np.mgrid[r0:r1, c0:c1]
            x, y = cols / cpf - half, half - rows / cpf
            self.allowed[r0:r1, c0:c1] &= (x - ox) ** 2 + (y - oy) ** 2 >= (c - EPS) ** 2
            for (dr, dc), table in zip(NEIGHBORS, edges):
                table[r0:r1, c0:c1] &= _edges_clear(x, y, dc / cpf, -dr / cpf, [(ox, oy, c)])
        self._steps = [(dr * n + dc, math.hypot(dr, dc) / cpf,
                        (table & self.allowed & _shifted(self.allowed, dr, dc)).ravel().tolist())
                       for (dr, dc), table in zip(NEIGHBORS, edges)]

    def point_clear(self, point) -> bool:
        x, y = point
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > self._limit or abs(y) > self._limit:
            return False
        return all((x - ox) ** 2 + (y - oy) ** 2 >= (c - EPS) ** 2 for ox, oy, c in self._obstacles)

    def segment_clear(self, start, end) -> bool:
        if not (self.point_clear(start) and self.point_clear(end)):
            return False
        span = math.dist(start, end)
        return all(abs(ox - start[0]) > c + span or abs(oy - start[1]) > c + span
                   or segment_distance((ox, oy), start, end) >= c - EPS
                   for ox, oy, c in self._obstacles)

    def _search(self, start, end_cost, exact_end=None):
        if not self.point_clear(start):
            raise ValueError("start_position_blocked")
        n = self.field.board_size
        costs, parents, heap = [math.inf] * (n * n), {}, []
        for (r, c), distance in self._nearby_cells(start):
            k = r * n + c
            costs[k], parents[k] = distance, None
            heapq.heappush(heap, (distance, k))
        best, finish = math.inf, None
        while heap:
            distance, k = heapq.heappop(heap)
            if distance > costs[k] + EPS:
                continue
            if distance > best + EPS:
                break
            cell = divmod(k, n)
            tail = end_cost(cell, self.field.cell_to_world(*cell))
            if tail is not None and distance + tail < best:
                best, finish = distance + tail, k
            for offset, step, clear in self._steps:
                if clear[k]:
                    other, candidate = k + offset, distance + step
                    if candidate + EPS < costs[other]:
                        costs[other], parents[other] = candidate, k
                        heapq.heappush(heap, (candidate, other))
        if finish is None:
            raise ValueError("target_unreachable")
        route = []
        while finish is not None:
            route.append(self.field.cell_to_world(*divmod(finish, n)))
            finish = parents[finish]
        route = [tuple(start), *reversed(route)]
        if exact_end is not None:
            route.append(tuple(exact_end))
        # Same shortcut smoothing as the engine.
        result, index = [route[0]], 0
        while index < len(route) - 1:
            following = len(route) - 1
            while following > index + 1 and not self.segment_clear(route[index], route[following]):
                following -= 1
            if math.dist(result[-1], route[following]) > EPS:
                result.append(route[following])
            index = following
        return result


class FieldStrategyEnv:
    distance = staticmethod(math.dist)

    def __init__(self, config: dict | None = None, opponent: str = "none",
                 seed: int = 0, opponent_predictor: Callable | None = None):
        self.cfg = config if config is not None else load_config(FIELD_CONFIG)
        if self.cfg.get("backend") != "field":
            raise ValueError("FieldStrategyEnv requires a config with backend='field'")
        if opponent not in OPPONENT_STYLES:
            raise ValueError(f"unsupported opponent: {opponent}")
        if opponent == "selfplay" and opponent_predictor is None:
            raise ValueError("selfplay requires a frozen opponent policy")
        self.codec = ActionCodec(self.cfg)
        self.encoder = StateEncoder(self.cfg)
        self.opponent = opponent
        self.opponent_predictor = opponent_predictor
        self.total_time = total_seconds(self.cfg)
        self.auto_time = float(self.cfg["autonomous_seconds"])
        self.endgame_start = self.total_time - float(self.cfg["endgame_seconds"])
        self.rules = self.cfg["override_rules"]
        self.points = {k: float(v) for k, v in self.cfg["score_points"].items()}
        self.half = float(self.cfg["field_env"]["field"]["half_width"])
        self.midfield = [tuple(map(float, p)) for p in self.rules["midfield_polygon"]]
        self.midfield_edges = list(zip(self.midfield, self.midfield[1:] + self.midfield[:1]))
        self.goal_spec = {g["id"]: g for g in self.cfg["goals"]}
        self.source_spec = {s["id"]: s for s in self.cfg["sources"]}
        self.toggle_spec = {t["id"]: t for t in self.cfg["toggles"]}
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None):
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        engine_cfg = deepcopy(self.cfg["field_env"])
        jitter = float(self.rules.get("spawn_jitter", 0.0))
        for robot in engine_cfg["robots"]:
            dx, dy = self.rng.uniform(-jitter, jitter, size=2) if jitter else (0.0, 0.0)
            robot["position"] = [robot["position"][0] + float(dx), robot["position"][1] + float(dy)]
        self.engine = GameEnv(config=engine_cfg)
        self.field = self.engine.field_state
        self.radius = float(self.field.settings["robot_radius"])
        if self.rules.get("preload", True):
            for robot in self.field.robots.values():
                robot.held_items[ITEM_TYPES.index(f"{TEAM_COLOR[robot.team]}_yellow")] = 1
        d = float(self.rules["midfield_staging_distance"])
        ring = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
                for r in (d, d + 0.55) for a in range(0, 360, 45)]
        self._staging = [p for p in ring if self._polygon_distance(p) <= self.radius - 0.05
                         and self.engine.allowance.point_clear(p)] or ring
        self.done = False
        self.auto_done = False
        self.auto_bonus = [0.0, 0.0]
        self.awp_met = [False, False]
        self.awp_snapshot = [0.0, 0.0]
        self.pending: dict[str, Pending | None] = {rid: None for rid in self.field.robots}
        self.stats = {key: [0, 0] for key in (
            "actions", "collected", "scored_pins", "scored_cups", "toggles_set", "failed_actions",
            "duplicate_targets", "rejected_actions", "collisions")}
        self.stats["wait_seconds"] = [0.0, 0.0]
        self.stats["trace"] = []
        self._version = 0
        self._cache: dict = {}
        self._collided: set[str] = set()
        self.active_robot = "A0"
        self._opponent_style = (self.rng.choice(["greedy", "coordination", "random"]).item()
                                if self.opponent == "mixed" else self.opponent)
        self._launch_opponents()
        return self.observation(), {"action_mask": self.action_mask(), "active_robot": self.active_robot}

    # ------------------------------------------------------------- utilities
    @property
    def time(self) -> float:
        return float(self.field.time)

    def _to_rl(self, xy) -> tuple[float, float]:
        return (float(xy[0]) + self.half, float(xy[1]) + self.half)

    @staticmethod
    def _team_color(team: int) -> str:
        return TEAM_COLOR[team]

    def _cached(self, key, build):
        """Memoize per engine tick and per dispatched action."""
        stamp = (self._version, self.field.step_count)
        hit = self._cache.get(key)
        if hit is None or hit[0] != stamp:
            hit = self._cache[key] = (stamp, build())
        return hit[1]

    def _polygon_distance(self, xy) -> float:
        if _point_in_polygon(xy, self.midfield):
            return 0.0
        return min(segment_distance(xy, a, b) for a, b in self.midfield_edges)

    def zone_of(self, xy) -> str:
        if self._polygon_distance(xy) < 1e-6:
            return "midfield"
        x, y = xy
        # Points on a diagonal go west/east by x sign so zones stay 180-degree symmetric.
        if abs(x) > abs(y) - 1e-6:
            return "west" if x < 0 else "east"
        return "north" if y > 0 else "south"

    def in_midfield(self, rid: str) -> bool:
        """SC6: any part of the circular robot is within the Midfield."""
        return self._polygon_distance(self.field.pose(rid).xy) <= self.radius

    def _midfield_counts(self) -> list[int]:
        counts = [0, 0]
        for rid, robot in self.field.robots.items():
            counts[robot.team] += self.in_midfield(rid)
        return counts

    # ---------------------------------------------------- item classification
    @staticmethod
    def pin_kinds(channel: int, team: int) -> tuple[str, ...]:
        """Strategy kinds a carried Pin can be scored as (which face shows)."""
        faces = PIN_FACES[channel]
        return tuple(kind for kind, color in (("alliance_pin", TEAM_COLOR[team]), ("yellow_pin", "yellow"))
                     if color in faces)

    @staticmethod
    def collectable_channels(kind: str, team: int) -> tuple[int, ...]:
        if kind == "cup":
            return (CUP,)
        color = TEAM_COLOR[team]
        if kind == "alliance_pin":  # any pin showing our color
            return tuple(i for i, faces in enumerate(PIN_FACES) if color in faces)
        # pins we would score yellow-up
        return tuple(i for i, faces in enumerate(PIN_FACES) if "yellow" in faces and color not in faces)

    def _carried(self, rid: str) -> tuple[int | None, bool]:
        held = self.field.robots[rid].held_items
        pins = np.flatnonzero(held[:CUP] > 0)
        return (int(pins[0]) if len(pins) else None), bool(held[CUP] > 0)

    def _carried_kind(self, rid: str) -> str | None:
        channel, _ = self._carried(rid)
        if channel is None:
            return None
        return "alliance_pin" if "alliance_pin" in self.pin_kinds(channel, self.field.robots[rid].team) else "yellow_pin"

    def _objects_by_zone(self) -> dict[str, list[dict]]:
        def build():
            zones = {z: [] for z in ZONES}
            for obj in self.field.object_states():
                zones[self.zone_of(obj["position"])].append(obj)
            return zones
        return self._cached("objects", build)

    def _zone_objects(self, zone: str, channels) -> list[dict]:
        return [o for o in self._objects_by_zone()[zone] if o["channel"] in channels]

    def _loader_channel(self, loader_id: str, kind: str, team: int) -> int | None:
        stock = self.field.loaders[loader_id].stock
        return next((ch for ch in self.collectable_channels(kind, team) if stock[ch] > 0), None)

    # --------------------------------------------------------------- scoring
    def _yellow_owner(self, goal, include_midfield: bool = False) -> int:
        quadrant = goal.quadrant if isinstance(goal, Goal) else self.goal_spec[goal]["quadrant"]
        if quadrant >= 0:
            owner = int(self.field.zone_colors[quadrant])
            return owner - 1 if owner else -1
        if include_midfield:
            red, blue = self._midfield_counts()
            return 0 if red > blue else 1 if blue > red else -1
        return -1

    def _visible_halves(self, goal_id: str, include_midfield: bool) -> tuple[list[int], list[int]]:
        """(colored, owned yellow) visible Pin halves per team."""
        colored, yellow = [0, 0], [0, 0]
        owner = None
        for channel, face in self.field.goals[goal_id].stack:
            if channel == CUP:
                continue
            if face == "yellow":
                if owner is None:
                    owner = self._yellow_owner(goal_id, include_midfield)
                if owner >= 0:
                    yellow[owner] += 1
            else:
                colored[TEAM_COLOR.index(face)] += 1
        return colored, yellow

    def _goal_halves(self, goal_id: str, include_midfield: bool = False) -> list[int]:
        colored, yellow = self._visible_halves(goal_id, include_midfield)
        return [colored[0] + yellow[0], colored[1] + yellow[1]]

    def scores(self, include_midfield: bool = False) -> list[float]:
        """Official-style match score; Midfield effects only at match end (SC1)."""
        result = list(self.auto_bonus)
        for goal_id in self.field.goals:
            colored, yellow = self._visible_halves(goal_id, include_midfield)
            for team in (0, 1):
                result[team] += self.points["alliance_colored_half"] * colored[team] + self.points["yellow_half"] * yellow[team]
        if include_midfield:
            for team, count in enumerate(self._midfield_counts()):
                result[team] += self.points["midfield_robot"] * count
        return result

    def _touching_perimeter(self, rid: str) -> bool:
        x, y = self.field.pose(rid).xy
        limit = self.half - self.radius - float(self.rules.get("perimeter_contact_margin", 0.05))
        return abs(x) >= limit or abs(y) >= limit

    def _awp_state(self, team: int) -> tuple[float, bool]:
        opposing = self.rules["home_quadrant"][TEAM_COLOR[1 - team]]
        halves = goals = 0
        for goal_id, spec in self.goal_spec.items():
            if spec["quadrant"] != opposing:
                count = self._goal_halves(goal_id, include_midfield=True)[team]
                halves += count
                goals += count >= 2
        awp = self.cfg["awp"]
        parts = (
            min(1.0, halves / max(1, int(awp["scored_halves"]))),
            min(1.0, goals / max(1, int(awp["goals_with_two_halves"]))),
            float(self._midfield_counts()[team] >= int(awp["required_midfield_robots"])),
            float(not awp["required_no_perimeter_contact"]
                  or not any(self._touching_perimeter(r) for r in (FRIENDLY, OPPONENTS)[team])),
        )
        return sum(parts) / 4, all(p >= 1 for p in parts)

    def _end_autonomous(self) -> float:
        """Award the Autonomous Bonus, freeze AWP, and return the AWP reward."""
        provisional = self.scores(include_midfield=bool(self.rules.get("autonomous_bonus_includes_midfield", True)))
        bonus = self.points["autonomous_bonus"]
        if provisional[0] == provisional[1]:
            self.auto_bonus = [x + bonus / 2 for x in self.auto_bonus]
        else:
            self.auto_bonus[int(provisional[1] > provisional[0])] += bonus
        reward = 0.0
        for team, sign in ((0, 1), (1, -1)):
            self.awp_snapshot[team], self.awp_met[team] = self._awp_state(team)
            reward += sign * float(self.cfg["reward"]["awp_bonus"]) * self.awp_met[team]
        self.auto_done = True
        return reward

    # ------------------------------------------------- strategy-level views
    @property
    def robots(self) -> dict[str, Robot]:
        def build():
            views = {}
            for rid, robot in self.field.robots.items():
                kind = self._carried_kind(rid)
                pin = f"{TEAM_COLOR[robot.team]}_pin" if kind == "alliance_pin" else kind
                views[rid] = Robot(rid, robot.team, self._to_rl(self.field.pose(rid).xy), pin,
                                   self._carried(rid)[1], self.pending[rid])
            return views
        return self._cached("robots", build)

    @property
    def goals(self) -> dict[str, Goal]:
        def build():
            views = {}
            for goal_id, spec in self.goal_spec.items():
                goal = self.field.goals[goal_id]
                stack = [StackItem("cup", -1) if channel == CUP else
                         StackItem(f"{face}_pin", TEAM_COLOR.index(face)) if face in TEAM_COLOR else
                         StackItem("yellow_pin", -1)
                         for channel, face in goal.stack]
                views[goal_id] = Goal(goal_id, tuple(spec["position"]), spec["quadrant"], spec["owner"],
                                      goal.max_items, stack)
            return views
        return self._cached("goals", build)

    @property
    def sources(self) -> dict[str, Source]:
        def build():
            views = {}
            for sid, spec in self.source_spec.items():
                available = {"red_pin": 0, "blue_pin": 0, "yellow_pin": 0, "cup": 0}
                if spec["type"] == "loader":
                    counts = enumerate(self.field.loaders[spec["loader"]].stock.tolist())
                else:
                    counts = ((o["channel"], o["count"]) for o in self._objects_by_zone()[spec["zone"]])
                for channel, n in counts:
                    if channel == CUP:
                        available["cup"] += n
                    elif PIN_FACES[channel] == ("yellow", "yellow"):
                        available["yellow_pin"] += n
                    else:
                        for color in set(PIN_FACES[channel]) & set(TEAM_COLOR):
                            available[f"{color}_pin"] += n
                views[sid] = Source(sid, tuple(spec["position"]), available)
            return views
        return self._cached("sources", build)

    @property
    def toggles(self) -> dict[str, int]:
        return {tid: int(self.field.zone_colors[t["quadrant"]]) - 1 for tid, t in self.toggle_spec.items()}

    def state(self, team: int = 0, active_robot: str | None = None) -> dict:
        our_ids, opponent_ids = (FRIENDLY, OPPONENTS) if team == 0 else (OPPONENTS, FRIENDLY)
        own, theirs = f"{TEAM_COLOR[team]}_pin", f"{TEAM_COLOR[1 - team]}_pin"
        now = self.scores(include_midfield=self.done)
        robots = {}
        for rid, view in self.robots.items():
            p = view.pending
            robots[rid] = {
                "position": view.position, "carried_pin_type": self._carried_kind(rid), "has_cup": view.cup,
                "current_target": p.action.target if p else None,
                "current_action": p.action.verb if p else None,
                "target_position": p.destination if p else view.position,
                "remaining_action_duration": max(0.0, p.finish_at - self.time) if p else 0.0,
                "busy": p is not None,
            }
        goals = []
        for goal_id, g in self.goals.items():
            counted = self._goal_halves(goal_id, include_midfield=self.done)
            kinds = [item.kind for item in g.stack]
            can_pin = g.can_add("red_pin")
            goals.append({
                "goal_id": goal_id, "position": g.position, "quadrant_id": g.quadrant, "owner": g.owner,
                "our_scored_pins": counted[team], "opponent_scored_pins": counted[1 - team],
                "yellow_pins": kinds.count("yellow_pin"), "our_alliance_pins": kinds.count(own),
                "opponent_alliance_pins": kinds.count(theirs), "cups": kinds.count("cup"),
                "stack_capacity": g.max_items, "remaining_capacity": g.max_items - len(g.stack),
                "can_place": can_pin, "can_place_pin": can_pin, "can_place_cup": g.can_add("cup"),
            })
        sources = [{
            "source_id": sid, "position": s.position,
            "available_alliance_pins": s.available[own], "available_opponent_pins": s.available[theirs],
            "available_yellow_pins": s.available["yellow_pin"], "available_cups": s.available["cup"],
            "available": dict(s.available),
        } for sid, s in self.sources.items()]
        toggle_owner = self.toggles
        toggles = [{"toggle_id": tid, "quadrant_id": t["quadrant"], "position": tuple(t["position"]),
                    "owner": ("neutral", "ours", "opponent")[0 if toggle_owner[tid] == -1 else
                                                             1 if toggle_owner[tid] == team else 2]}
                   for tid, t in self.toggle_spec.items()]
        phase = ("autonomous" if not self.auto_done else
                 "endgame" if self.time >= self.endgame_start else "driver")
        return {
            "time": self.time, "time_remaining": max(0.0, self.total_time - self.time), "phase": phase,
            "score_difference": now[team] - now[1 - team], "our_score": now[team], "opponent_score": now[1 - team],
            "awp_progress": self.awp_snapshot[team] if self.auto_done else self._awp_state(team)[0],
            "awp_met": self.awp_met[team],
            "our_robot_ids": our_ids, "opponent_robot_ids": opponent_ids, "active_robot": active_robot,
            "robots": robots, "goals": goals, "toggles": toggles, "sources": sources,
        }

    def observation(self, team: int = 0, active_robot: str | None = None) -> np.ndarray:
        return self.encoder.encode(self.state(team, active_robot or self.active_robot), team)

    # ------------------------------------------------------- legality / mask
    def _goal_reserved(self, goal_id: str, by: str) -> bool:
        return any(rid != by and p is not None and p.action.verb == "SCORE" and p.action.target == goal_id
                   for rid, p in self.pending.items())

    def _legal(self, rid: str, act: Action, team: int, channel: int | None, cup: bool) -> bool:
        if act.verb == "COLLECT":
            if cup if act.object_type == "cup" else channel is not None:
                return False  # SG6: at most one Pin and one Cup
            spec = self.source_spec[act.target]
            if spec["type"] == "zone":
                return bool(self._zone_objects(spec["zone"], self.collectable_channels(act.object_type, team)))
            loader = self.field.loaders[spec["loader"]]
            return ((loader.team is None or loader.team == team)
                    and (self.time >= self.auto_time or not self.rules["matchload_driver_only"])
                    and self._loader_channel(spec["loader"], act.object_type, team) is not None)
        if act.verb == "SCORE":
            spec = self.goal_spec[act.target]
            if spec["owner"] not in ("neutral", TEAM_COLOR[team]):
                return False  # SG9: opposing Alliance Goals are protected
            if act.object_type == "cup":
                if not cup or not self.goals[act.target].can_add("cup"):
                    return False
            elif channel is None or act.object_type not in self.pin_kinds(channel, team) \
                    or not self.goals[act.target].can_add("red_pin"):
                return False
            if self._goal_reserved(act.target, rid):
                return False
            return not (spec["quadrant"] < 0 and self.rules["no_midfield_goal_in_endgame"]
                        and self.time + self.action_duration(rid, act) >= self.endgame_start)  # SG12
        if act.verb == "SET_TOGGLE":
            return self.toggles[act.target] != team
        if act.verb == "GO_MIDFIELD":
            return not self.in_midfield(rid)
        return act.verb == "WAIT"  # BLOCK is reserved

    def action_mask(self, robot_id: str | None = None) -> np.ndarray:
        rid = robot_id or self.active_robot
        if rid is None or self.done or self.pending[rid] is not None:
            return np.zeros(len(self.codec), dtype=np.bool_)

        def build():
            team = self.field.robots[rid].team
            channel, cup = self._carried(rid)
            return np.fromiter((self._legal(rid, act, team, channel, cup) for act in self.codec.actions),
                               dtype=np.bool_, count=len(self.codec))
        return self._cached(("mask", rid), build).copy()

    # ------------------------------------------- strategy -> native actions
    def _nearest_object(self, rid: str, zone: str, channels) -> dict | None:
        here = self.field.pose(rid).xy
        return min(self._zone_objects(zone, channels), key=lambda o: math.dist(here, o["position"]), default=None)

    def _midfield_staging(self, rid: str) -> tuple[float, float]:
        def build():
            here = self.field.pose(rid).xy
            others = [self.field.pose(o).xy for o in self.field.robots if o != rid]
            clear = [p for p in self._staging if all(math.dist(p, o) >= 2 * self.radius + 0.05 for o in others)]
            return min(clear or self._staging, key=lambda p: math.dist(here, p))
        return self._cached(("staging", rid), build)

    def destination(self, robot_id: str, action: Action) -> tuple[float, float]:
        """Strategy-frame (0..12) target position, for estimates and features."""
        if action.verb == "COLLECT":
            spec = self.source_spec[action.target]
            if spec["type"] == "zone":
                team = self.field.robots[robot_id].team
                obj = self._nearest_object(robot_id, spec["zone"], self.collectable_channels(action.object_type, team))
                if obj is not None:
                    return self._to_rl(obj["position"])
            return tuple(spec["position"])
        if action.verb == "SCORE":
            return tuple(self.goal_spec[action.target]["position"])
        if action.verb == "SET_TOGGLE":
            return tuple(self.toggle_spec[action.target]["position"])
        if action.verb == "GO_MIDFIELD":
            return self._to_rl(self._midfield_staging(robot_id))
        return self._to_rl(self.field.pose(robot_id).xy)

    def action_duration(self, robot_id: str, action: Action) -> float:
        """Straight-line estimate used by masks and heuristics only."""
        if action.verb == "WAIT":
            return float(self.cfg["action_seconds"]["wait"])
        start = self._to_rl(self.field.pose(robot_id).xy)
        travel = math.dist(start, self.destination(robot_id, action)) / float(self.cfg["robot_speed_units_per_second"])
        return travel + float(self.cfg["action_seconds"][action.verb.lower()])

    def to_native(self, rid: str, action: Action) -> NativeAction:
        team = self.field.robots[rid].team
        if action.verb == "COLLECT":
            spec = self.source_spec[action.target]
            if spec["type"] == "loader":
                channel = self._loader_channel(spec["loader"], action.object_type, team)
                return NativeAction("MATCHLOAD", target=spec["loader"], item=ITEM_TYPES[channel])
            obj = self._nearest_object(rid, spec["zone"], self.collectable_channels(action.object_type, team))
            return NativeAction("COLLECT", target=obj["object_id"])
        if action.verb == "SCORE":
            if action.object_type == "cup":
                return NativeAction("SCORE", target=action.target, item="cup")
            face = TEAM_COLOR[team] if action.object_type == "alliance_pin" else "yellow"
            return NativeAction("SCORE", target=action.target, item=ITEM_TYPES[self._carried(rid)[0]], face=face)
        if action.verb == "SET_TOGGLE":
            return NativeAction("TOGGLE", target=action.target)
        if action.verb == "GO_MIDFIELD":
            return NativeAction("MOVE", position=self._midfield_staging(rid))
        if action.verb == "WAIT":
            return NativeAction("WAIT")
        raise NotImplementedError("BLOCK is reserved and always masked")

    def _start_native(self, robot_id: str, native: NativeAction):
        """GameEnv.start_action, but routed around the other robots."""
        here = self.field.pose(robot_id).xy
        obstacles = []
        for rid in self.field.robots:
            if rid != robot_id:
                there = self.field.pose(rid).xy
                # A robot we already touch may be driven away from, never through.
                obstacles.append((there, min(2 * self.radius + 0.05, math.dist(here, there) - 0.01)))
        task = prepare_action(self.field, RobotAwareAllowance(self.field, obstacles), robot_id, native)
        if robot_id in self._collided and task.moving:
            # Random 0-3 tick back-off breaks symmetric collision livelocks.
            ticks = int(self.rng.integers(0, 4))
            task.poses = (task.poses[0],) * ticks + task.poses
            task.estimated_duration += ticks * float(self.field.settings["seconds_per_step"])
        self._collided.discard(robot_id)
        self.field.robots[robot_id].task = task
        return task

    def _dispatch(self, robot_id: str, action_id: int) -> bool:
        action_id = int(action_id)
        if not 0 <= action_id < len(self.codec) or not self.action_mask(robot_id)[action_id]:
            raise ValueError(f"illegal action {action_id} for {robot_id} at t={self.time:.3f}")
        team = self.field.robots[robot_id].team
        action = self.codec.decode(action_id)
        duplicate = bool(action.target) and any(
            rid != robot_id and self.field.robots[rid].team == team and p is not None
            and (p.action.verb, p.action.target) == (action.verb, action.target)
            for rid, p in self.pending.items())
        self.stats["duplicate_targets"][team] += duplicate
        destination = self.destination(robot_id, action)
        try:
            task = self._start_native(robot_id, self.to_native(robot_id, action))
        except ValueError:
            # Planner refused (e.g. unreachable target): spend a WAIT so time advances.
            self.stats["rejected_actions"][team] += 1
            self.stats["failed_actions"][team] += 1
            action, destination = Action("WAIT"), self._to_rl(self.field.pose(robot_id).xy)
            task = self.engine.start_action(robot_id, NativeAction("WAIT"))
        self.pending[robot_id] = Pending(action, self.time + task.estimated_duration, destination)
        self.stats["actions"][team] += 1
        self.stats["trace"].append((round(self.time, 3), robot_id, str(action), round(task.estimated_duration, 3)))
        self._version += 1
        return duplicate

    def _launch_opponents(self):
        if self.opponent == "none" or self.done:
            return
        for rid in OPPONENTS:
            if self.pending[rid] is not None:
                continue
            if self._opponent_style == "selfplay":
                mask = self.action_mask(rid)
                obs = self.observation(team=1, active_robot=rid)
                canonical = int(self.opponent_predictor(obs, self.codec.canonical_mask(mask, team=1)))
                choice = self.codec.physical_index(canonical, team=1)
            else:
                choice = choose_action(self, rid, self._opponent_style, self.rng)
            self._dispatch(rid, choice)

    def _record(self, results: dict):
        stat_for = {"COLLECT": "collected", "SET_TOGGLE": "toggles_set"}
        for rid, result in results.items():
            status = result["status"]
            if status == "running":
                continue
            team = self.field.robots[rid].team
            pending = self.pending[rid]
            if status == "complete" and pending is not None:
                verb = pending.action.verb
                if verb == "SCORE":
                    self.stats["scored_cups" if pending.action.object_type == "cup" else "scored_pins"][team] += 1
                elif verb == "WAIT":
                    self.stats["wait_seconds"][team] += float(self.cfg["action_seconds"]["wait"])
                elif verb in stat_for:
                    self.stats[stat_for[verb]][team] += 1
            elif status == "failed" and result.get("reason") != "match_ended_before_completion":
                self.stats["failed_actions"][team] += 1
                if result.get("reason") == "motion_collision":
                    self.stats["collisions"][team] += 1
                    self._collided.add(rid)
            self.pending[rid] = None

    def _next_friendly_idle(self) -> str | None:
        return next((rid for rid in FRIENDLY if self.pending[rid] is None), None)

    def _stats_info(self, dt: float) -> dict:
        s = self.scores(include_midfield=self.done)
        info = {
            "delta_time": dt, "game_time": self.time, "active_robot": self.active_robot,
            "our_score": s[0], "opponent_score": s[1], "score_difference": s[0] - s[1],
            "win": (1.0 if s[0] > s[1] else 0.0 if s[0] < s[1] else 0.5) if self.done else None,
            "awp_ours": self.awp_met[0], "awp_opponent": self.awp_met[1],
            "awp_progress": self.awp_snapshot[0] if self.auto_done else self._awp_state(0)[0],
        }
        for key in ("duplicate_targets", "wait_seconds", "scored_pins", "scored_cups", "collected",
                    "toggles_set", "failed_actions", "rejected_actions", "collisions"):
            info[key] = self.stats[key][0]
        return info

    # ------------------------------------------------------------------ step
    def step(self, action_id: int):
        if self.done or self.active_robot is None:
            raise RuntimeError("episode finished; call reset()")
        reward_cfg = self.cfg["reward"]
        robot_id = self.active_robot
        duplicate = self._dispatch(robot_id, action_id)
        penalty = float(reward_cfg["duplicate_target_penalty"]) * duplicate
        if self.pending[robot_id].action.verb == "WAIT":
            penalty += float(reward_cfg["wait_penalty_per_second"]) * float(self.cfg["action_seconds"]["wait"])
        next_idle = self._next_friendly_idle()
        if next_idle is not None:
            self.active_robot = next_idle
            return self.observation(), -penalty, False, False, self._stats_info(0.0)

        t_start = self.time
        old = self.scores()
        reward = -penalty
        while True:
            _, _, engine_done, info = self.engine.step()
            self._version += 1
            self._record(info["results"])
            if not self.auto_done and self.time >= self.auto_time - 1e-9:
                reward += self._end_autonomous()
            if engine_done:
                self.done, self.active_robot = True, None
                break
            self._launch_opponents()
            self.active_robot = self._next_friendly_idle()
            if self.active_robot is not None:
                break
        new = self.scores(include_midfield=self.done)
        diff = new[0] - new[1]
        reward += (diff - (old[0] - old[1])) * float(reward_cfg["score_difference_scale"])
        if self.done:
            reward += float(reward_cfg["terminal_win_bonus"]) * float(np.sign(diff))
        obs = self.encoder.empty() if self.done else self.observation()
        return obs, float(reward), self.done, False, self._stats_info(self.time - t_start)
