# Strategy RL ↔ Full GameEnv Integration Contract (share with teammate)

Our RL code can train now using an **abstract standalone environment**. The teammate's full environment should later implement the **same interface** so we can swap its dynamics without changing PPO, baselines or evaluation.

## Required methods

```python
reset(seed: int | None = None) -> tuple[np.ndarray, dict]
action_mask(robot_id: str | None = None) -> np.ndarray  # bool, shape [num_actions]
step(action_id: int) -> tuple[np.ndarray, float, bool, bool, dict]
state(team: int = 0, active_robot: str | None = None) -> dict
```

Fields:
- `.active_robot`: ID of the next *friendly* robot to act (`"A0"` / `"A1"`), or `None` after terminal.
- `.encoder.obs_dim`: fixed state-vector dimension. Current version: 284 floats.
- `.codec`: same ordered `ActionCodec(config)`; 55 possible action IDs, `BLOCK` masked.
- `reset` returns state vector and `{"action_mask": ..., "active_robot": ...}`.
- `step` returns `(next_state_vector, reward, terminated, truncated, info)`.
- `info["delta_time"]`: **actual simulated seconds elapsed** since this RL decision (may be 0 for second simultaneous friendly action).
- At terminal, `active_robot=None`, fixed-shape zero observation is okay, and final `info` contains scores/win/efficiency metrics.

A Protocol and check are defined in `vex_rl/contract.py`, so `check_contract(adapter_instance)` provides a basic shape/flow verification. Do not assume replacing variable names alone guarantees semantic compatibility.

## Observations (`state()` dictionary)

See `StrategyEnv.state()` in `vex_rl/simulator.py`. Key names must match:

```
{
  "time_remaining": float,
  "phase": "autonomous" | "driver" | "endgame",
  "score_difference": float,
  "awp_progress": float,
  "our_score": float,
  "opponent_score": float,
  "our_robot_ids": ("A0", "A1"),
  "opponent_robot_ids": ("B0", "B1"),
  "active_robot": "A0" | "A1" | None,
  "robots": {id: {
     "position": (x,y), "carried_pin_type": None | "alliance_pin" | "yellow_pin",
     "has_cup": bool, "busy": bool, "current_target": str | None,
     "current_action": str | None, "target_position": (x,y),
     "remaining_action_duration": float
  }},
  "goals": [{
     "goal_id": str, "position": (x,y), "quadrant_id": int,
     "owner": "red" | "blue" | "neutral", "our_scored_pins": int,
     "opponent_scored_pins": int, "yellow_pins": int, "cups": int,
     "stack_capacity": int, "remaining_capacity": int,
     "can_place_pin": bool, "can_place_cup": bool
  }],
  "toggles": [{"toggle_id": str, "quadrant_id": int,
     "position": (x,y), "owner": "ours" | "neutral" | "opponent"}],
  "sources": [{"source_id": str, "position": (x,y),
     "available_alliance_pins": int, "available_opponent_pins": int,
     "available_yellow_pins": int, "available_cups": int}]
}
```

`StateEncoder` also relies on `time`, `awp_met` and goal fields `our_alliance_pins`, `opponent_alliance_pins`, `can_place`. The complete, authoritative schema for the current encoder is in `features.py` and `simulator.state()`.

Coordinates: normalized to a 12x12 field model in `config.json`. Units must be consistent with movement speed and duration. The model does not assume physics-level camera/collision coordinates.

## Actions

The source / goal / toggle order and IDs must match **the config's ordered arrays**. `config.json -> symmetry` maps canonical red policy actions to their blue physical counterparts using a 180-degree rotation. Preserve that mapping (or revise it deliberately) for reliable self-play. Action IDs:

- `COLLECT(source_id, alliance_pin|yellow_pin|cup)`
- `SCORE(goal_id, alliance_pin|yellow_pin|cup)`
- `SET_TOGGLE(toggle_id)`
- `GO_MIDFIELD`
- `WAIT`
- `BLOCK(location)` reserved for future and **always illegal** at present.

Action masking should account for available resources, cargo capacity (one Pin and one Cup per robot), legal target, pending actions, score capacity, time, and game-specific constraints. A mask must *never* have all entries false for an idle robot: keep WAIT valid.

## Time & race conditions

Friendly robots may execute their macro actions concurrently. The central policy chooses for any *currently idle* friendly robot. The simulator then advances to the next completion event; it must return the elapsed simulated seconds in `info.delta_time`. Match: 30 s autonomous + 90 s driver (2026–27 VEX U). Opponent decisions are scheduled separately by the GameEnv. Store each robot's `current_target`, `remaining_action_duration`, and `busy` status.

Avoid double-consuming the last Source object or scoring into the same capped Goal at the same instant: add reservations / atomic final-state validation. The example simulator reserves source stock at action scheduling, and blocks overlapping Goal scoring actions.

## Reward, match termination & evaluation

Keep official **game points** separate from heuristic **training reward**. Current reward is based on change in *team score difference*, AWP bonus, terminal victory, small penalties for duplicate target/WAIT (in `config.json`). Do not apply the AWP Win Point as match score. `reset` starts a fresh match and ends at `t=120.0`; `step(...).terminated=True` ends an episode.

The RL training loop applies time-aware GAE using elapsed seconds from `info.delta_time`. If the new simulator changes the reward convention, performance numbers across versions are not directly comparable and the model should be retrained/re-evaluated.

## Suggested integration sequence

1. Map teammate GameEnv raw state → `state()` schema (`StateEncoder` stays unchanged).
2. Map ActionCodec index → native GameEnv macro action, validating legality via native rules.
3. Map completion timestamps / robot busy states → current API.
4. Compute game scores / AWP / endgame and reward consistently.
5. Run `check_contract`; then extend tests to compare both implementations on a fixed small set of deterministic scenarios.
6. Train and evaluate PPO **on the integrated environment**; do not assume the surrogate checkpoint transfers directly.

The actual geometry and official-rule scoring of the teammate's environment can be more detailed; the current standalone stack/field arrangement is explicitly a research approximation.
