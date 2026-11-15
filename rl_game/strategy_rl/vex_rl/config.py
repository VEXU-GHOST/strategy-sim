"""Validated, explicit game/training parameters.

Two backends share this loader:

* ``"backend": "surrogate"`` (default) - the abstract, event-driven strategy
  simulator in ``simulator.py``. Goals/sources/toggles are listed directly.
* ``"backend": "field"`` - the teammate's tick-based field engine
  (``rl_game/env/vex_field_environment``) wrapped by ``field_backend.py``.
  The engine layout lives under ``field_env``; the strategy-level ``goals``,
  ``toggles``, ``initial_robots`` and source positions are *derived* from it
  here, so field coordinates have a single source of truth.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.json"
FIELD_CONFIG = ROOT / "config_override_field.json"
ZONE_CENTROIDS = {"west": (-4.0, 0.0), "north": (0.0, 4.0), "east": (4.0, 0.0),
                  "south": (0.0, -4.0), "midfield": (0.0, 0.0)}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    filename = Path(path) if path is not None else DEFAULT_CONFIG
    with filename.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("backend", "surrogate") == "field":
        data = derive_field_sections(data)
    validate_config(data)
    return data


def derive_field_sections(data: dict) -> dict:
    """Fill strategy-level sections from the field engine layout.

    Engine coordinates are centered feet (-6..6); strategy coordinates are the
    surrogate's 0..field_size frame, so the PPO encoder and its 180-degree
    symmetry are unchanged.
    """
    data = deepcopy(data)
    engine = data["field_env"]
    half = float(engine["field"]["half_width"])
    if abs(2 * half - float(data["field_size"])) > 1e-9:
        raise ValueError("field_size must equal 2 * field_env.field.half_width")

    def to_rl(p):
        return [float(p[0]) + half, float(p[1]) + half]

    layout = engine["layout"]
    env_settings = engine.setdefault("environment", {})
    # One clock: the engine episode is exactly autonomous + driver.
    env_settings["episode_seconds"] = float(data["autonomous_seconds"]) + float(data["driver_seconds"])
    env_settings.setdefault("wait_seconds", data["action_seconds"]["wait"])
    data["robot_speed_units_per_second"] = float(env_settings.get("robot_speed", 3.0))

    data["goals"] = []
    for goal in layout["goals"]:
        if not isinstance(goal, dict) or "id" not in goal:
            raise ValueError("field backend goals must be dicts with explicit ids")
        quadrant = goal.get("quadrant")
        data["goals"].append({
            "id": goal["id"], "position": to_rl(goal["position"]),
            "quadrant": -1 if quadrant is None else int(quadrant),
            "owner": goal.get("owner", "neutral"),
            "max_stack_items": int(goal.get("capacity", env_settings.get("goal_capacity", 10))),
        })
    data["toggles"] = [{"id": t["id"], "position": to_rl(t["position"]), "quadrant": int(t["quadrant"])}
                       for t in layout.get("toggles", [])]
    data["initial_robots"] = {r["id"]: to_rl(r["position"]) for r in engine["robots"]}
    loaders = {entry["id"]: entry for entry in layout.get("matchloaders", [])}
    for source in data["sources"]:
        if source["type"] == "loader":
            loader = loaders.get(source["loader"])
            if loader is None:
                raise ValueError(f"source {source['id']} names unknown loader {source['loader']}")
            source["position"] = to_rl(loader["position"])
        elif source["type"] == "zone":
            if source["zone"] not in ZONE_CENTROIDS:
                raise ValueError(f"unknown zone {source['zone']}")
            source["position"] = to_rl(ZONE_CENTROIDS[source["zone"]])
        else:
            raise ValueError(f"source type must be loader or zone: {source}")
    return data


def validate_config(data: dict) -> None:
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
    stock_keys = ("cup", "yellow_pin", "red_pin", "blue_pin")
    if any(s.get(k, 0) < 0 for s in data["sources"] for k in stock_keys):
        raise ValueError("source stocks must be nonnegative")
    if any(g["max_stack_items"] < 1 for g in data["goals"]):
        raise ValueError("goals must have positive stack capacity")
    for item in (*data["goals"], *data["toggles"], *data["sources"]):
        if not all(math.isfinite(float(v)) for v in item["position"]):
            raise ValueError(f"non-finite position in {item['id']}")


def total_seconds(config: dict) -> float:
    return float(config["autonomous_seconds"] + config["driver_seconds"])
