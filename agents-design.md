# Agent Design Log

## 2026-03-08: Restructure to PettingZoo ParallelEnv (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Restructure `ghost_strategy` package into PettingZoo-convention `push_back` environment.

**Changes**:
- Renamed package `ghost_strategy/` → `push_back/`
- Renamed `env/env.py` → `env/push_back.py`
- Created `push_back/push_back_v0.py` (version-control entry point with `env()` / `parallel_env()` factories)
- Refactored `GhostStrategyEnv` → `PushBackEnv(ParallelEnv)`:
  - `step(actions)` now takes `dict[str, np.ndarray]` (agent_name → [dx, dy, dheading])
  - Returns `(observations, rewards, terminations, truncations, infos)` per PettingZoo API
  - `reset(seed, options)` returns `(observations, infos)`
  - Added `observation_space()` (Dict with balls/agents/score) and `action_space()` (Box)
  - Agents no longer passed into env constructor — actions come externally via `step()`
- Updated `cli.py` to bridge `Agent.act()` → numpy array → `env.step()` 
- Made `pettingzoo` and `gymnasium` required dependencies (no longer optional)
- Updated all imports from `ghost_strategy` → `push_back`

**File structure**:
```
push_back/
    __init__.py
    cli.py
    env/
        __init__.py
        push_back.py          # PushBackEnv(ParallelEnv)
        state.py              # WorldState, Pose
        render.py             # render_state()
        agents.py             # Action, Agent protocol, RandomAgent, StandStill
```

_(push_back_v0.py was removed immediately after — unnecessary indirection for early dev.)_

---

## 2026-03-08: Discretize to grid world (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Discretize all positions to a 4" grid, heading to 8 directions, and actions to discrete moves.

### Design discussion (chat log)

**User**: "the default timestep should be 0.1sec, 60sec total. All measurements should be in inches, I want pose and any movements discretized to 4inch blocks."

**Copilot** asked clarifying questions:
1. Should actions be discrete (N/S/E/W/stay) or continuous-then-snapped?
2. Heading discretization: 4 or 8 directions?
3. Speed: 1 cell per tick max?

**User**: "discrete state. The robot movement will be limited (it's a diff drive), make a function that will allow some movements from one pose (x,y,head) to another. Limit heading to 8 ways, I know the distance is inaccurate, but add a comment and hand-wave that away, our robot drivetrain is much faster than we will limit it to in this sim anyway. Yeah do heading faces direction of movement, robot has a turn primitive to turn 45deg in 3 timesteps, either direction. make sure this is a configurable value (angular velocity discretized)."

### Changes

**Constants** (in `state.py`):
- `CELL_SIZE = 4` inches per grid cell
- `FIELD_INCHES = 144`, `GRID_SIZE = 36` (positions 0..35)
- `DT = 0.1` s/tick, `DEFAULT_MAX_STEPS = 600` (60 s total)
- `DEFAULT_TURN_TICKS = 3` (configurable)

**Heading** — `IntEnum` with 8 values counterclockwise from East:
  E=0, NE=1, N=2, NW=3, W=4, SW=5, S=6, SE=7

**HEADING_DELTAS** — grid (dx,dy) per heading. Diagonal moves cover 4√2≈5.66" not 4"; accepted per user ("drivetrain is much faster than we limit it").

**Action** — moved from `agents.py` to `state.py` as `IntEnum`:
  STAY=0, FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3

**`apply_action(pose, action, turn_progress, turn_direction, turn_ticks)`**:
- FORWARD: move 1 cell in facing direction (clamped to grid bounds)
- TURN_LEFT/RIGHT: accumulate `turn_progress`; heading changes after `turn_ticks` consecutive same-direction turns; switching direction or other action resets progress
- STAY: no-op, resets turn progress

**Pose**: `x: int, y: int, heading: int` (grid coords + heading index)

**PushBackEnv**:
- `action_space` → `Discrete(4)` (was `Box(3,)`)
- `observation_space` → all `int32` (was `float64`)
- `step(actions: dict[str, int])` — per-agent turn tracking via `_turn_progress` / `_turn_direction` lists
- Truncation at `max_steps` (600 by default)
- `turn_ticks` configurable in constructor

**Agents**: `Agent.act()` returns `int` (was `Action` dataclass). `RandomAgent` picks uniformly from 4 actions. `StandStill` returns `Action.STAY`.

**Renderer**: `_to_px()` converts grid→pixels via `gx * CELL_SIZE * PPI`. Heading line uses `heading * π/4` radians.

**CLI**: Default 600 steps. Actions are plain ints now (no numpy conversion needed).

---

## 2026-03-08: Heterogeneous agents + macro-actions (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Support heterogeneous agents — some emit primitives, some emit high-level "go to goal" macro-actions. Restructure agents into separate files.

### Design discussion (chat log)

**User**: "make an agents directory inside the env directory, and move agents, one agent per file there. move actions there as well. I am going to have heterogeneous agents, how does pettingzoo handle that?"

**Copilot**: PettingZoo handles this natively — `action_space(agent)` is per-agent. Argued `Action` should stay in `state.py` (it's a sim primitive used by `apply_action`) to avoid circular imports.

**User**: "some of my agents can only make longer horizon moves (Go To Goal1 goal2, etc), and some other agents can only make forward, left, right moves."

**Copilot** proposed 3 options:
- **A: Macro-actions / options framework** — agent picks a goal, planner generates primitive queue, resolver drains one per tick. While busy, agent's action is ignored.
- **B: Two-level action space** — agent picks goal or CONTINUE each tick, enabling mid-move cancellation/replanning.
- **C: Different tick rates** — high-level acts every N ticks.

Recommended B for MCTS, but user wanted A.

**User**: "I want option a. put option b in a comment saying we might wanna support cancellation and mid-move replanning later down the line"

### Architecture

```
push_back/env/
    agents/
        __init__.py          # re-exports
        protocol.py          # Agent protocol + ActionResolver ABC
                             #   PrimitiveResolver (identity)
                             #   MacroResolver (queues planned primitives)
        stand_still.py       # StandStill (returns STAY)
        random_agent.py      # RandomAgent (uniform random primitive)
        high_level.py        # GoToAgent (picks goal index)
    planner.py               # Greedy grid pathfinder: pose + goal → deque[primitive]
    push_back.py             # PushBackEnv — per-agent ActionResolver
    state.py                 # Action enum stays here (sim primitive)
```

### Key design: ActionResolver (options framework)

- `ActionResolver` is an ABC with `resolve(action, pose) → int`, `busy() → bool`, `reset()`.
- `PrimitiveResolver`: identity pass-through for low-level agents.
- `MacroResolver(goals, turn_ticks)`: takes a list of `(gx, gy)` goal cells. When action index `i` is chosen, `planner.plan()` generates a deque of primitives. Drains one per tick. **While busy, incoming actions are ignored** (options framework semantics).
- TODO comment in protocol.py re: Option B for mid-move cancellation.

### PushBackEnv changes

- Constructor takes optional `resolvers: dict[str, ActionResolver]`. Agents without a resolver get `PrimitiveResolver` (backward compatible).
- `action_space(agent)` returns `Discrete(len(goals))` for MacroResolver agents, `Discrete(4)` for primitive agents.
- `step()` routes each action through the agent's resolver before `apply_action()`.
- `reset()` calls `resolver.reset()` on all resolvers.

### Planner

- Greedy: align heading to target, walk straight. Each 45° turn costs `turn_ticks` primitives.
- No obstacle avoidance yet (agents don't block each other). Comment suggests A* if needed later.

### CLI

- Demo: red agents use `GoToAgent` + `MacroResolver` (4 goal cells), blue agents use `RandomAgent` + `StandStill` with default `PrimitiveResolver`.

### Old agents.py

- Deleted. Agent protocol and implementations now live in `robots/` subpackage.

---

## 2026-03-08: BaseRobot refactor (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Refactor from separate Agent + ActionResolver to unified BaseRobot class hierarchy where each robot type owns its observation_space, action_space, tick (decision), and resolve (action→primitive).

### Design decisions

- **`BaseObservation`**: frozen dataclass holding `balls`, `agents`, `score` arrays
- **`BaseAction`**: frozen dataclass wrapping a `gymnasium.Space`
- **`BaseRobot` (ABC)**: abstract methods `observation_space(n_balls)`, `action_space()`, `tick(obs, agent_id)→int`, `resolve(action, pose)→int`; optional overrides `reset()`, `busy()`; static `build_obs(state)`
- **`StandStill`** and **`RandomAgent`**: primitive robots — `resolve()` is identity (passes action through)
- **`GoToRobot`**: absorbs old `GoToAgent` + `MacroResolver` into one class. Owns the goal list, maintains a primitive queue, and uses the planner to convert goal selection → primitives

### Changes

- Created `push_back/env/robots/base.py` with `BaseObservation`, `BaseAction`, `BaseRobot`
- Rewrote `stand_still.py`, `random_agent.py`, `high_level.py` as `BaseRobot` subclasses
- Deleted `protocol.py` (Agent protocol, ActionResolver, PrimitiveResolver, MacroResolver all absorbed)
- Updated `PushBackEnv`: constructor takes `robots: dict[str, BaseRobot]` instead of `resolvers`; delegates `observation_space()` and `action_space()` to robot instances
- Updated `cli.py`: builds `robots` dict directly, calls `robot.tick()` in step loop
- Updated all `__init__.py` exports

### File structure

```
push_back/env/robots/
    __init__.py          # re-exports
    base.py              # BaseObservation, BaseAction, BaseRobot (ABC)
    stand_still.py       # StandStill(BaseRobot) — concrete base for all robots
    random_robot.py      # RandomRobot(StandStill)
    high_level.py        # GoToRobot(StandStill)
push_back/env/
    field.py             # DEFAULT_GOALS
```

---

## 2026-03-08: StandStill as concrete base + field.py + RandomRobot rename (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Make `StandStill` the concrete base class for all robots, move goals to `field.py`, rename `RandomAgent` → `RandomRobot`.

### Changes

- **`field.py`** created — holds `DEFAULT_GOALS: list[tuple[int, int]]`
- **`StandStill`** is now the concrete base class: provides default `observation_space`, `action_space`, `tick` (STAY), and `resolve` (identity). All other robots inherit from it
- **`RandomRobot`** (renamed from `RandomAgent`): extends `StandStill`, only overrides `tick`
- **`GoToRobot`**: extends `StandStill` instead of `BaseRobot`; `goals` defaults to `DEFAULT_GOALS` from `field.py`; inherits `observation_space` from `StandStill`
- `BaseRobot` ABC remains as the type-annotation interface

### Inheritance

```
BaseRobot (ABC)
  └── StandStill          ← concrete defaults (STAY, identity resolve)
        ├── RandomRobot   ← overrides tick
        └── GoToRobot     ← overrides action_space, tick, resolve, busy, reset
```

## 2026-03-08: Add goals (Copilot)





## 2026-03-08: Render goals and ball slots (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Add goal tube and ball-slot rendering to `render.py`.

**Changes**:
- `_draw_goal()` helper: draws a gray line between `interface_a` and `interface_b`, then spaces slot circles evenly along it — filled slots get their `BallColor`, empty slots are hollow gray outlines.
- `render_state()` draws goals between the field border and loose balls (goals behind balls in z-order).

## 2026-03-08: Collision system + action masking (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Add collision boundaries for walls and goals; prevent robots from crossing them.

**Changes**:
- `state.py`: Added `ROBOT_DIA = 4` (grid cells), `ROBOT_RADIUS = 2`; `WorldState` now carries `collision_segments` and `blocked_cells`.
- `field.py`: Added `CollisionSegment` NamedTuple, `GOAL_WIDTH_IN = 5.53`, `make_collision_segments()` (4 wall segments + 4-segment rectangle per goal, 36 total).
- `collision.py` (new): `compute_blocked_cells()` marks grid cells within `ROBOT_RADIUS * CELL_SIZE` inches of any segment; `resolve_moves()` rejects blocked-cell moves and robot-robot overlaps (both revert).
- `push_back.py`: `reset()` builds collision data; `step()` collects proposed poses, validates via `resolve_moves()`.
- `render.py`: Draws collision segments from `state.collision_segments`; `AGENT_RADIUS` derived from `ROBOT_RADIUS`.
- `random_robot.py`: `valid_actions()` masks FORWARD when destination is blocked; `tick()` samples from valid actions only. Old bounds check removed (env handles centrally).
- `base.py`: Added `set_blocked_cells()` for action masking support.
