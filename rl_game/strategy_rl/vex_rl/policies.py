"""Simple rules baselines, always filtered by the environment's action mask."""
from __future__ import annotations
import numpy as np


def choose_action(env, robot_id: str, style: str, rng: np.random.Generator) -> int:
    mask = env.action_mask(robot_id)
    available = np.flatnonzero(mask)
    if not len(available):
        raise RuntimeError("no legal actions")
    if style == "random":
        return int(rng.choice(available))
    if style not in {"greedy", "coordination"}:
        raise ValueError(f"unknown heuristic: {style}")

    robot = env.robots[robot_id]
    team = robot.team
    rest = env.total_time - env.time
    teammate_targets = {
        (mate.pending.action.verb, mate.pending.action.target)
        for mate in env.robots.values()
        if mate.team == team and mate.name != robot_id and mate.pending is not None
    }
    rng_tiebreak = rng.uniform(0, 0.001, size=len(env.codec))
    best_index, best_value = None, -float("inf")
    for idx in available:
        a = env.codec.decode(int(idx))
        duration = env.action_duration(robot_id, a)
        if duration >= rest + 0.2 and a.verb != "WAIT":
            value = -5.0  # likely unable to finish before the match ends
        elif a.verb == "SCORE":
            if a.object_type == "alliance_pin":
                value = 9.0 / (duration + 0.6)
            elif a.object_type == "yellow_pin":
                yellow_owner = env._yellow_owner(env.goals[a.target])
                value = (15 if yellow_owner == team else 4.0) / (duration + 0.7)
            else:
                # Cups enable the next pin, and have no direct scoring points.
                value = (6.0 if rest > 8.0 else 0.8) / (duration + 1.0)
        elif a.verb == "COLLECT":
            if a.object_type == "alliance_pin":
                nearest_scoring_goal = min(
                    (env.distance(env.sources[a.target].position, goal.position)
                     for goal in env.goals.values()
                     if goal.owner in ("neutral", env._team_color(team)) and goal.can_add("red_pin")),
                    default=8.0)
                value = 8.0 / (duration + 0.6 + nearest_scoring_goal / env.cfg["robot_speed_units_per_second"])
            elif a.object_type == "yellow_pin":
                viable = any(env._yellow_owner(goal) == team and goal.can_add("yellow_pin")
                             for goal in env.goals.values())
                value = (11.0 if viable else 3.0) / (duration + 1.0)
            else:
                unfinished = any(goal.can_add("cup") and
                                 goal.owner in ("neutral", env._team_color(team))
                                 for goal in env.goals.values())
                value = (6.0 if unfinished else 1.0) / (duration + 1.0)
        elif a.verb == "SET_TOGGLE":
            quadrant = next(t["quadrant"] for t in env.cfg["toggles"] if t["id"] == a.target)
            yellow_count = sum(sum(item.kind == "yellow_pin" for item in goal.stack)
                               for goal in env.goals.values() if goal.quadrant == quadrant)
            old_owner = env.toggles[a.target]
            swing = 10.0 * yellow_count * (2.0 if old_owner == 1 - team else 1.0)
            value = (swing + 0.75) / (duration + 0.8)
        elif a.verb == "GO_MIDFIELD":
            if rest <= 15.0:
                value = (14.0 if rest <= 8.0 else 10.0) / (duration + 0.5)
            elif env.time < env.auto_time and env.auto_time - env.time <= 6.0:
                value = 4.5 / (duration + 0.6)
            else:
                value = 0.25 / (duration + 1.0)
        elif a.verb == "WAIT":
            value = -0.2
        else:
            value = -1.0
        if style == "coordination" and (a.verb, a.target) in teammate_targets and a.target:
            value -= 8.0
        value += float(rng_tiebreak[idx])
        if value > best_value:
            best_index, best_value = int(idx), value
    assert best_index is not None and mask[best_index]
    return best_index
