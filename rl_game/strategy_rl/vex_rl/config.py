"""Validated, explicit game/training parameters, independent of any teammate GameEnv."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    filename = Path(path) if path is not None else DEFAULT_CONFIG
    with filename.open(encoding="utf-8") as f:
        data = json.load(f)
    if float(data["robot_speed_units_per_second"]) <= 0:
        raise ValueError("robot_speed_units_per_second must be positive")
    if not 0 < float(data["autonomous_seconds"]) < float(data["autonomous_seconds"]) + float(data["driver_seconds"]):
        raise ValueError("game time must be positive")
    if any(float(t) <= 0 for t in data["action_seconds"].values()):
        raise ValueError("action_seconds must be positive")
    for key in ("goals", "toggles", "sources"):
        values = data[key]
        if not values or len({r["id"] for r in values}) != len(values):
            raise ValueError(f"{key} must have unique nonempty IDs")
    if set(data["initial_robots"]) != {"A0", "A1", "B0", "B1"}:
        raise ValueError("expected robot IDs A0 A1 B0 B1")
    if len(data["toggles"]) != 4 or len(data["goals"]) != 9:
        raise ValueError("this version expects 4 toggles and 9 goals")
    for group in ("goals", "sources", "toggles"):
        ids = {item["id"] for item in data[group]}
        smap = data["symmetry"][group]
        if set(smap) != ids or set(smap.values()) != ids:
            raise ValueError(f"symmetry.{group} must be a permutation of IDs")
        if any(smap[smap[key]] != key for key in ids):
            raise ValueError(f"symmetry.{group} must be involutive")
    if any(s["cup"] < 0 or s["yellow_pin"] < 0 or
           s["red_pin"] < 0 or s["blue_pin"] < 0 for s in data["sources"]):
        raise ValueError("source stocks must be nonnegative")
    if any(g["max_stack_items"] < 1 for g in data["goals"]):
        raise ValueError("goals must have positive stack capacity")
    return deepcopy(data)


def total_seconds(config: dict) -> float:
    return float(config["autonomous_seconds"] + config["driver_seconds"])
