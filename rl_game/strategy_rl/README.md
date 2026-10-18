# VEX U Override — High-Level Strategy RL (Centralized Masked PPO)

Independent, runnable Python strategy-simulation + RL pipeline. You **do not** need the teammate's unfinished `GameEnv` to start experimenting. Once their implementation is ready, implement the adapter contract described in `TEAMMATE_INTERFACE.md` and re-train/re-evaluate on the integrated simulator.

**This is a research strategy surrogate, NOT an exact official gameplay/physics simulator.** Read the `GAME_ASSUMPTIONS.md` before interpreting performance numbers.

## One-minute start (PowerShell / VSCode)

From your existing `strategy-sim` folder, download `VEX_Strategy_RL_Full.zip` into that folder and run:

```powershell
Expand-Archive -Path ".\VEX_Strategy_RL_Full.zip" -DestinationPath "." -Force
cd ".\rl_game\strategy_rl"
python -m pip install -r requirements.txt
python run.py test
python run.py smoke
python run.py baselines --episodes 30 --opponent greedy
python run.py curriculum --steps-per-stage 20000 --eval-episodes 20 --out-dir outputs/curriculum
```

If `python` doesn't work on your Windows installation, use `py` (or your Conda environment's Python). Existing `rl_game/v1`, `agents`, `env`, `models`, and `training` are NOT modified by the ZIP.

### Evaluate / plot

```powershell
python run.py eval --policy ppo --model outputs/curriculum/ppo_selfplay.pt --opponent greedy --episodes 50 --csv outputs/eval_greedy.csv
python run.py eval --policy coordination --opponent greedy --episodes 50 --csv outputs/eval_coordination.csv
python run.py plot --histories outputs/curriculum/history_no_opponent.csv outputs/curriculum/history_greedy.csv outputs/curriculum/history_mixed.csv outputs/curriculum/history_selfplay.csv --out outputs/training_curve.png
python run.py demo --policy coordination --opponent greedy --limit 40
```

A quick initial test: `python run.py curriculum --steps-per-stage 1000 --eval-episodes 3 --bc-episodes 2 --out-dir outputs/quickcheck`. **A quick test proves the pipeline works; it does not demonstrate convergence or performance superiority.** For the real experiment use more training steps, evaluation episodes, and multiple seeds.

## What is implemented

| Module | Responsibility |
|--|--|
| `config.json` | Editable game length, action/movement durations, initial positions, goals, resources, rewards, PPO hyperparameters |
| `vex_rl/actions.py` | Stable, explicit 55-action discrete macro-action catalogue |
| `vex_rl/types.py` | Robot / Goal / Source / PendingAction dataclasses |
| `vex_rl/simulator.py` | Full event-driven match transitions; sources/collect/score/cups/toggles/midfield; scores; AWP proxy; action mask; opponent scheduling |
| `vex_rl/features.py` | Team-relative fixed-size state vector (284 floats with default config) |
| `vex_rl/policies.py` | Random, greedy, coordination baselines |
| `vex_rl/model.py` | Centralized shared actor-critic network; masked action probabilities; safe checkpoint handling |
| `vex_rl/trainer.py` | PPO, variable-time advantage computation, behavior-cloning warm-start, staged curriculum, frozen snapshot self-play |
| `vex_rl/evaluate.py` | Reproducible per-episode CSV, score/win-rate/AWP/coordination statistics, training plot |
| `vex_rl/contract.py` | Stable environment interface for teammate integration |
| `tests/` | Unit tests for actions, score, masking, time, terminals, checkpoint, PPO |

## State and action specification

**Global:** `time_remaining`, `phase`, `score_difference`, `awp_progress` / `awp_met`.

**Robots:** two ours + two opponents, positions, carried pin type, has cup, busy/target/time-left.

**Goals (9):** owner, quadrant, visible scored halves, yellow pins, cup count, stack capacity/availability.

**Toggles (4):** team 0 / neutral / team 1.

**Sources (6 by default):** position, available red/blue/yellow pins and cups. Sources in the config are configurable and abstracted; they are *not* claimed to match exact official field spawn locations.

**Actions (55 with defaults):**
- `COLLECT(source_id, alliance_pin/yellow_pin/cup)` = 6×3 = 18 actions.
- `SCORE(goal_id, alliance_pin/yellow_pin/cup)` = 9×3 = 27 actions.
- `SET_TOGGLE(toggle_id)` = 4 actions.
- `GO_MIDFIELD` = 1.
- `WAIT` = 1.
- `BLOCK(location)` = 4 **reserved & always masked** for a future stage.

Masks enforce inventory, finite source stock, allowed goal ownership, alternating Pin→Cup→Pin stacking, and no simultaneous scoring into one goal. Two teammates can collect or set toggles concurrently, and their duplicate-target attempts are tracked.

**Symmetric self-play:** Blue-side PPO observations are rotated 180 degrees and goal/source/toggle feature order is remapped with `config.json -> symmetry`. Predicted action IDs and masks are also remapped, so the same learned policy can control red and blue teams. If the field layout/IDs change, update this explicit symmetry mapping and retrain as needed.

## One centralized policy, two asynchronously acting robots

At `reset`, the central policy selects an action for A0. Next it selects an action for A1, at the same simulation time (`delta_time = 0`). After both are busy, the simulator advances time to the next action-completion event or autonomous/episode boundary. As soon as one friendly robot is free, the policy makes its next decision. B0/B1 are handled by the opponent controller.

`WAIT` is guaranteed available to idle robots, so there are no all-false masks. `BLOCK` remains masked. The action timers are derived from movement distance/speed plus a per-action overhead. **No low-level robotics or collision physics is simulated.**

## RL reward versus official game score

The simulator computes a **game score** separately from the **learning reward**. Game scores use official point magnitudes for visible colored/yellow pin halves, the autonomous bonus, and midfield endgame robots, with a deliberately simplified stacking/visibility model.

The default shaped reward is

```
R = (new_team_score_difference - old_team_score_difference) * 0.1
    + 3.0 * (our_AWP_achieved - their_AWP_achieved) at autonomous cutoff
    + 4.0 * sign(final_team_score_difference) at the final whistle
    - 0.02 * WAIT_seconds
    - 0.04 * duplicate_target_attempts
```

AWP is a **Win Point**, not extra match score. AWP is measured at 30 seconds and held fixed. Game score's autonomous bonus is awarded according to VEX U's autonomous scoring, including midfield at the cutoff (modelled approximately).

**Time-aware PPO:** each transition's discount is `gamma_per_second ** delta_time`; GAE uses `(gamma_per_second*lambda_per_second)**delta_time`. There is no extra discount for the zero-simulated-time A0/A1 scheduling transition.

## Curriculum

1. `no_opponent`: optional coordination teacher BC warm-start, then PPO versus idle opponents.
2. `greedy`: PPO against greedy enemy policies.
3. `mixed`: PPO versus per-episode random/greedy/coordination opponents.
4. `selfplay`: PPO versus a **frozen snapshot** from stage 3. This is snapshot self-play, not continuously updated league self-play.

Each stage saves a `.pt` checkpoint, a `history_*.csv`, and evaluation scores vs 4 opponent types. `curriculum_summary.csv` contains the stage results, and model inference is deterministic by default during evaluation.

## Tunable action timing (requested)

Edit `config.json` before training:

```json
"robot_speed_units_per_second": 4.0,
"action_seconds": {
  "collect": 0.8,
  "score": 1.0,
  "set_toggle": 0.7,
  "go_midfield": 0.5,
  "wait": 0.75
}
```

An action's duration is `EuclideanDistance(robot,target) / robot_speed_units_per_second + action_seconds[type]`, except `WAIT`, which uses its fixed time. You can change these without modifying Python. Changing action/observation ordering requires updating the adapter and retraining; changing timing values also warrants re-evaluation and often re-training.

## Interpreting evaluation

Compare agents against the **same opponent, scenario, config, episode seeds and episode count**. Win rate is 0.5 for a tie. Report mean/final score, win rate, autonomous win point, duplicate targeting, WAIT seconds, pins/cups scored. Include several training seeds and confidence intervals before claiming PPO improves on coordination heuristics. Scores are **surrogate environment results**, not real VEX competition predictions.

## Official references

- [VEX Override v2.1 game rules](https://manuals.vex.com/v5/override/v2.1/game/)
- [VEX U modifications VUG5/VUG6/VUT4/VUT5](https://manuals.vex.com/v5/override/v2.1/vex-u/)
- [Official VEX 2026–27 competition page](https://www.vexrobotics.com/v5/competition)

Validated with Python 3.13, PyTorch 2.10 (CPU), NumPy 2.3. The project supports modern Python 3.10+ and uses only NumPy/PyTorch for learning; Matplotlib is only needed for PNG plots.
