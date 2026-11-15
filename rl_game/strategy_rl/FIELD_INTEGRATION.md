# Strategy RL ↔ field environment integration (VEX Override 2026–27)

The centralized Masked PPO in `vex_rl/` now trains directly on the tick-based field
engine in `rl_game/env/vex_field_environment`. The engine source is **not modified**.
The abstract surrogate (`simulator.py`, `config.json`) still works and is kept for
fast experiments and regression tests.

```powershell
cd rl_game\strategy_rl
python run.py test                       # surrogate + field-backend tests
python run.py smoke                      # field backend is the default (--env field)
python run.py baselines --episodes 20 --opponent greedy
python run.py curriculum --steps-per-stage 20000 --eval-episodes 20 --out-dir outputs/field_curriculum
python run.py --env surrogate smoke      # the old abstract simulator
```

## Layers

| Layer | File | Responsibility |
| --- | --- | --- |
| Field engine (Marcus) | `rl_game/env/vex_field_environment/*` | Geometry, Dijkstra routes, constant-rate motion, collisions, cargo, goal stacks, Toggle/Loader interactions, 0.25 s clock |
| Override rules + adapter | `vex_rl/field_backend.py` | Rules the engine leaves out, 64 macro actions → native actions, semi-MDP stepping, PPO contract |
| Layout + rules config | `config_override_field.json` | `field_env` is the engine config verbatim; RL goals/toggles/sources/starts are derived from it in `vex_rl/config.py` |
| Backend switch | `vex_rl/envs.py` | `make_env(config)` picks `StrategyEnv` or `FieldStrategyEnv` from `config["backend"]` |

PPO, the heuristics (`policies.py`), evaluation and the curriculum are unchanged apart
from calling `make_env`.

## Override rules implemented in the adapter

From the Override v2.1 manual text:

- 9 Goals: 2 red and 2 blue Alliance Goals, 4 short neutral Goals, 1 tall Midfield Goal.
  Opposing Alliance Goals are masked (SG9). No Midfield-Goal placement that would finish in the Endgame (SG12).
- Possession: at most one Pin and one Cup (SG6), via the engine's cargo slots.
- One Pin preload per robot (red/yellow for red, blue/yellow for blue).
- Match Loads only in the Driver period, only from your own Alliance's Loaders.
- Scoring: 5 per visible alliance-colored half, 10 per *owned* yellow half. Quadrant yellow ownership
  comes from that Quadrant's Toggle. Midfield-Goal yellow ownership goes to the Alliance with more robots
  in the Midfield at match end (SC5). 8 points per robot with any part in the Midfield at match end (SC6).
  Autonomous Bonus 12 (6/6 on a tie) (SC7). AWP is tracked separately and never added to match score.
- Scores that depend on the Midfield are counted only at the final whistle (SC1). During the match the
  learning signal uses the provisional score.

## Action space (64 IDs, fixed order from the config)

- `COLLECT(source, alliance_pin|yellow_pin|cup)` over 9 sources:
  - `L_RN, L_RS, L_BN, L_BS`: Loaders → native `MATCHLOAD`
  - `Z_W, Z_N, Z_E, Z_S, Z_MID`: floor zones (triangular Quadrants + Midfield) → native `COLLECT` of the
    nearest matching loose object in that zone
- `SCORE(goal, kind)` over 9 goals → native `SCORE` with the face chosen by `kind`. A red/yellow pin can be
  scored red-up (`alliance_pin`, 5 points) or yellow-up (`yellow_pin`, 10 points if the Toggle is yours).
  Choosing the face is part of the policy.
- `SET_TOGGLE(T_W|T_N|T_E|T_S)` → native `TOGGLE`
- `GO_MIDFIELD` → native `MOVE` to the nearest free staging point inside the Midfield
- `WAIT` (engine `wait_seconds` = 0.75), plus 4 `BLOCK` IDs that are reserved and always masked

"alliance_pin" means any pin showing your color (red/yellow, red/blue). "yellow_pin" means a pin you would
score yellow-up (yellow/yellow, or the opponent's color/yellow).

## Stepping and timing

Each PPO decision dispatches one idle friendly robot. If the other friendly robot is idle too, the env
returns immediately with `delta_time = 0`. Otherwise it calls `GameEnv.step()` tick by tick, launching
opponent actions whenever B0/B1 go idle, until a friendly robot finishes or fails. Then it returns the
elapsed seconds for the time-aware GAE.

Robot interaction is handled as follows:

- The engine fails a task when robots would collide. The adapter plans each new task with an
  `AllowanceMap` subclass that also treats the other robots' **current** poses as obstacles. This prevents
  re-planning straight through a stationary robot.
- After a collision, the robot's next task starts with a random 0–3 tick back-off. This breaks symmetric
  livelocks, where two robots would otherwise replan into each other every tick.
- If the native planner rejects an action (for example, the target is unreachable), the robot spends a WAIT
  instead. It is counted in `failed_actions` and `rejected_actions`.

## Coordinates and teams

The engine uses centered feet (-6..6). The PPO encoder uses the surrogate's 0..12 frame (`+6`), so the
180° self-play symmetry is unchanged. In this config **A0/A1 are red** (team 0, west side) and **B0/B1
are blue**. That is the reverse of the engine's own `field_config.json`, where A is blue. Quadrant IDs:
0 = west, 1 = north, 2 = east, 3 = south. They rotate as `q → (q+2) % 4`.

## Known gaps / TODO before trusting numbers

1. **Coordinates are a draft.** Goal, Toggle and Loader positions come from the manual's prose: 8 goals
   one tile from one wall and two tiles from another, Toggles at wall centers, Loaders at the corners.
   Cups and corner pins come from the engine's `field_config.json` video reference. The Midfield polygon
   comes from that file's diagonal endpoints. Replace them with official field-drawing values, then retrain.
   `test_blue_view_is_symmetric_at_reset` checks that the layout stays 180°-symmetric.
2. Object counts are lower than the official ones (63 Pins / 56 Cups). Only a subset is placed.
3. Goal capacities (7 short / 11 tall) and timing are carried over from the surrogate assumptions.
4. Not modelled: autonomous-line crossing violations, robot pushing, descoring, removing objects from your
   own goals, Toggle "three orientations", and DROP/BLOCK strategy actions.
5. Periods are VEX U (30 s autonomous + 90 s driver), matching the existing strategy config. V5RC uses
   15 s + 1:45. Change `autonomous_seconds`/`driver_seconds` if you want V5RC timing.
6. Checkpoints trained on `config.json` (surrogate) do **not** load on this backend (302-dim obs, 64 actions
   vs 284/55). Train new ones.
