"""Team-relative, fixed-width observation vectors; no dependency on simulator internals."""
from __future__ import annotations
import numpy as np

PHASES = ("autonomous", "driver", "endgame")
ACTION_TYPES = ("COLLECT", "SCORE", "SET_TOGGLE", "GO_MIDFIELD", "WAIT", "BLOCK")


def one_hot(value, options):
    return [float(value == x) for x in options]


class StateEncoder:
    def __init__(self, config: dict):
        self.cfg = config
        self.field = float(config["field_size"])
        self.total = float(config["autonomous_seconds"] + config["driver_seconds"])
        self.symmetry = config["symmetry"]
        self.obs_dim = 11 + 4 * 16 + len(config["goals"]) * 17 + len(config["sources"]) * 6 + len(config["toggles"]) * 5

    def pos(self, position, team: int):
        x, y = float(position[0]), float(position[1])
        # 180-degree rotation: blue policy sees its own side in the same
        # orientation as the red policy, including mirrored goal/source IDs.
        if team == 1:
            x, y = self.field - x, self.field - y
        return [x / self.field, y / self.field]

    def encode(self, state: dict, team: int) -> np.ndarray:
        ours, enemies = state["our_robot_ids"], state["opponent_robot_ids"]
        active = state["active_robot"]
        phase = state["phase"]
        features = [
            state["time"] / self.total,
            state["time_remaining"] / self.total,
            float(phase == "autonomous"), float(phase == "endgame"),
            state["score_difference"] / 100.0,
            state["our_score"] / 100.0,
            state["opponent_score"] / 100.0,
            state["awp_progress"], float(state["awp_met"]),
            float(active == ours[0]), float(active == ours[1]),
        ]
        for rid in (*ours, *enemies):
            robot = state["robots"][rid]
            features.extend(self.pos(robot["position"], team))
            features.extend(one_hot(robot["carried_pin_type"], (None, "alliance_pin", "yellow_pin")))
            features.extend([
                float(robot["has_cup"]), float(robot["busy"]),
                robot["remaining_action_duration"] / max(1.0, self.total),
            ])
            features.extend(self.pos(robot["target_position"], team))
            features.extend(one_hot(robot["current_action"], ACTION_TYPES))
        goals_by_id = {goal["goal_id"]: goal for goal in state["goals"]}
        for spec in self.cfg["goals"]:
            goal_id = (self.symmetry["goals"][spec["id"]] if team == 1 else spec["id"])
            goal = goals_by_id[goal_id]
            features.extend(self.pos(goal["position"], team))
            quadrant = goal["quadrant_id"]
            if team == 1 and quadrant >= 0:
                quadrant = (quadrant + 2) % 4
            features.extend(one_hot(quadrant, (-1, 0, 1, 2, 3)))
            relative_owner = ("neutral" if goal["owner"] == "neutral" else
                              "ours" if goal["owner"] == ("red" if team == 0 else "blue") else "opponent")
            features.extend(one_hot(relative_owner, ("ours", "neutral", "opponent")))
            features.extend([
                goal["our_scored_pins"] / 10.0,
                goal["opponent_scored_pins"] / 10.0,
                goal["yellow_pins"] / 10.0,
                goal["cups"] / 10.0,
                goal["remaining_capacity"] / max(1, goal["stack_capacity"]),
                float(goal["can_place_pin"]),
                float(goal["can_place_cup"]),
            ])
        sources_by_id = {source["source_id"]: source for source in state["sources"]}
        for spec in self.cfg["sources"]:
            source_id = (self.symmetry["sources"][spec["id"]] if team == 1 else spec["id"])
            source = sources_by_id[source_id]
            features.extend(self.pos(source["position"], team))
            features.extend([
                source["available_alliance_pins"] / 10.0,
                source["available_opponent_pins"] / 10.0,
                source["available_yellow_pins"] / 10.0,
                source["available_cups"] / 10.0,
            ])
        toggles_by_id = {toggle["toggle_id"]: toggle for toggle in state["toggles"]}
        for spec in self.cfg["toggles"]:
            toggle_id = (self.symmetry["toggles"][spec["id"]] if team == 1 else spec["id"])
            toggle = toggles_by_id[toggle_id]
            features.extend(self.pos(toggle["position"], team))
            features.extend(one_hot(toggle["owner"], ("ours", "neutral", "opponent")))
        obs = np.asarray(features, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise AssertionError(f"observation dimension mismatch: actual={obs.shape}, expected={self.obs_dim}")
        if not np.isfinite(obs).all():
            raise ValueError("invalid observation NaN/Inf")
        return obs

    def empty(self) -> np.ndarray:
        return np.zeros(self.obs_dim, dtype=np.float32)
