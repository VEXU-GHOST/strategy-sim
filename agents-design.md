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

## 2026-03-08: Ball Ingestion & Spit Mechanic (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Changes**:
- `state.py`: Added `HEADING_DELTAS` (canonical location), `agent_color()` helper, `robot_held_balls: list[list[BallColor]]` to `WorldState`.
- `push_back.py`: After `resolve_moves()`, checks each robot's front cell (ROBOT_RADIUS cells ahead in heading direction) for balls. Ingested balls go through `robot.filter_balls()` — kept balls stored in `robot_held_balls`, spat balls placed at back cell (ROBOT_RADIUS cells behind).
- `base.py`: Added `filter_balls(balls, own_color) -> (keep, spit)` — default keeps all.
- `random_robot.py`: Overrides `filter_balls()` to keep own-color, spit opponent-color. Imports `HEADING_DELTAS` from `state.py` instead of local duplicate.
- `render.py`: Balls aggregated by cell — single ball = circle, multiple same-color = circle + count, mixed red/blue = pie-split with counts. Robots display held ball counts (red left, blue right) inside circle.

## 2026-03-08: SweeperRobot (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Changes**:
- `robots/sweeper.py` (new): Boustrophedon (lawn-mower) sweep across the field. Generates row endpoints as waypoints (alternating E/W), navigates with cardinal turns. Stuck detection: skips waypoint after 4 ticks of FORWARD-but-didn't-move (handles robot-robot collision rejection by `resolve_moves`). Inherits kinematics + `filter_balls` from `RandomRobot`.
- `robots/__init__.py`: Registered `SweeperRobot`.
- `cli.py`: blue_0 now uses `SweeperRobot` (was `RandomRobot`).

## 2026-03-08: Replace CLI with run scripts (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Changes**:
- `push_back/runner.py` (new): Shared `run_sim()` and `save_frames()` utilities.
- `runs/default.py` (new): Self-contained run script replacing the typer CLI. Configure robots, balls, steps, etc. as plain Python.
- `pyproject.toml`: Removed `[project.scripts]` entry and `typer` dependency.
- `cli.py`: Kept for reference but no longer the entry point.

## 2026-03-15: CLI Enhancements & Ball Override (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Changes**:
- `cli.py`: Added `--balls` flag to pass ball positions as `"x,y,color;..."`. Dual output mode: `.gif` → GIF only, `.png` → numbered PNGs only, no extension → both. Added `tqdm` progress bars for simulation and rendering. Robot lineup changed to StandStill (red) + SweeperRobot (blue_0).
- `push_back.py`: `reset()` accepts `options={"balls": np.ndarray}` to override random ball placement. Prints ball IDs (index, position, color) at start.
- `render.py`: Fixed heading line direction (pixel axes swapped vs grid axes). Larger label font (`size=20` with fallback). Axis labels drawn last so they render on top.

## 2026-03-15: Project Config & Docs (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Changes**:
- `pyproject.toml`: Renamed project to `ghost-strategy`, entry point → `ghost-strategy`. Removed `pettingzoo`/`gymnasium` from core deps, added `tqdm`. Optional `[zoo]` extra for `pettingzoo`.
- `.gitignore`: Added `outputs` directory.
- `README.md`: Updated CLI usage examples for new dual-output format.
- `AGENTS.md`: Added minimal quick-start documentation convention.

## 2026-03-15: mpv docs & remove PNG output (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Changes**:
- `README.md`: Added "Viewing Output" section documenting `mpv` keybinds (frame stepping, speed, pause) and commands for GIF/image-sequence/frame-overlay playback.
- `cli.py`: Removed PNG output path. `--out` now always produces a GIF (`.gif` appended if missing). Removed dual-output mode.
- `runner.py`: `save_frames()` always saves GIF; removed PNG branch.
- `README.md`: Updated CLI usage comments to reflect GIF-only output.

## 2026-03-15: Sweeper stuck-recovery improvement (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Fix sweeper boustrophedon algorithm's stuck recovery — previously skipped entire rows when hitting obstacles near row endpoints.

**Changes** (two iterations):
1. *First attempt (broken)*: Inserted transition waypoints at `(current_x, next_row_y)` on stuck. This caused an infinite insertion loop — if the transition point was also unreachable, each tick inserted another copy and advanced `wp_index`, bloating the list to 800+ entries while the robot sat still.
2. *Final fix*: Reverted stuck-recovery to simple `_wp_index += 1`. Instead changed `_generate_waypoints()` to emit **two waypoints per row** (start + end) so each skip loses at most half a row. Bumped `_STUCK_THRESHOLD` from 1→3 to reduce premature skips.

---

## 2026-03-15: Render pipeline optimization (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Optimize render + encode pipeline. Started at ~13 ms/frame render + sequential ffmpeg encode.

### Optimization rounds

| Round | Change | Before | After |
|-------|--------|--------|-------|
| 1 | Cache fonts at module level (`_FONT_SM/MD/LG`); use `stroke_width` param instead of 28-call outline hack | 19.3 ms/frame | 6.7 ms/frame |
| 2 | Cache static background (grid, border, blocked cells, collision segments, goals, labels) — reuse via `.copy()` | 6.7 ms/frame | 5.6 ms/frame |
| 3 | LRU-cached robot sprites (`_make_agent_sprite`); extracted HUD panels to `render_hud.py` with `@lru_cache` | 5.6 ms/frame | 3.0 ms/frame |
| 4 | Eliminated `np.array(img)` conversion — return PIL `Image` directly from `render()`, pipe `img.tobytes()` to ffmpeg | 4.6 ms/frame* | 2.4 ms/frame |
| 5 | Replaced manual `_bg_cache` globals with `@lru_cache(maxsize=2)` on `_render_background` using hashable geometry args (`frozenset`, tuples) | — | Code cleanup (same perf) |
| 6 | Streaming ffmpeg: start encoder before sim loop, pipe each frame immediately after rendering (overlap encode with next render) | 6.8s total | **4.4s total** |
| 7 | 1 MB pipe buffer (`fcntl F_SETPIPE_SZ`) — reduces backpressure from ffmpeg | — | Slight reduce in encode blocking |
| 8 | Threaded writer: `_Encoder` class with `queue.Queue(maxsize=30)` and background writer thread — encoding fully overlapped with rendering | 4.4s total | **4.0s total** (drain ≈0.15s) |
| 9 | PPI 5→3: halved resolution (field 720→432 px). Fonts scaled proportionally (`max(8, int(12*PPI/5))` etc.) | 4.0s total | **1.6s total** (2.5×) |
| 10 | Per-side margins: replaced uniform `IMG_MARGIN=25*PPI` with `MARGIN_LEFT/TOP/RIGHT/BOTTOM`. Image no longer square — sized to content. | 582×582 | 508×480 (−32% pixels) |
| 11 | HUD sidebar: scaled `render_hud.py` font/line-height with PPI. Moved HUD panels into left margin (no field overlap). | HUD overlaps field | Clean sidebar |
| 12 | PPI 3→6: doubled resolution for legibility while keeping previous optimizations | 1.6s total | **2.2s total** (1128×960) |
| 13 | Drain timer: explicit `drain` field in timing output showing `enc.finish()` wait time | — | Observability |
| 14 | Dead code removal: deleted `runner.py` (`run_sim`/`save_frames`) and `runs/default.py` — CLI supersedes both | — | Code cleanup |

*\*Round 4 "before" measured total render including np.array that was previously uncaptured.*

### Granular timing instrumentation

Added per-phase timing accumulators in `render.py` (`_timings` dict) with `get_render_timings()`. Phases tracked: `bg_hash`, `bg_copy`, `balls`, `agents`, `hud_step`, `hud_robot`, `hud_ball`, `tobytes`. CLI prints breakdown after simulation.

### Key findings

- `bg_copy` (PIL `Image.copy()`) is the dominant render cost — irreducible since we need a clean canvas for dynamic elements each frame.
- `np.array(img)` was costing 2.1 ms/frame (65% of render time before removal). PIL's `.tobytes()` is ~0 ms — C-level memcpy vs numpy overhead.
- ffmpeg preset comparison (9 configs tested): spread was narrow (4.2–5.9s for 901 frames). `veryfast` chosen — only 0.3s slower than `ultrafast` with much better compression (538K vs 1.5MB).
- Streaming + threaded ffmpeg overlaps encoding with rendering. Queue depth of 30 frames smooths bursts; drain time is typically <0.2s.
- Per-side margins saved ~32% pixel area (PPI=3) by not wasting space on sides with no content.
- At PPI=6 the whole pipeline (900 steps, 901 frames, 1128×960) runs in ~2.2s wall-clock.

### Files changed

- `render.py`: Font caching, background `@lru_cache`, sprite `@lru_cache`, per-phase timing, per-side margins (`MARGIN_LEFT/TOP/RIGHT/BOTTOM`), PPI-scaled fonts, removed dead code (`_draw_goal`, manual cache globals).
- `render_hud.py` (new): `render_step_panel`, `render_robot_panel`, `render_ball_panel` — all `@lru_cache`-wrapped, font/line-height scaled with PPI.
- `push_back.py`: `render()` returns `PIL.Image.Image` instead of `np.ndarray`.
- `cli.py`: `_Encoder` class (threaded ffmpeg writer with `queue.Queue`), 1 MB pipe buffer, streaming encode, per-phase timing breakdown + drain timer.
- `runner.py`: **Deleted** — `run_sim()`/`save_frames()` superseded by CLI streaming encoder.
- `runs/default.py`: **Deleted** — legacy entrypoint superseded by `push-back` CLI.

---

## 2026-03-16: HUD lru_cache pyramid + encode_to_file experiment (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

### HUD cache pyramid refactor

**Task**: Replace bespoke module-level glyph pre-rendering with a clean `@lru_cache` pyramid.

**Before**: `render_hud.py` had imperative module-level code eagerly building `_DIGIT_GLYPHS` list and `_PREFIX_IMG` at import time. `render_robot_panel` and `render_ball_panel` used `draw.text()` directly (slow on cache miss).

**After**: Three-layer `@lru_cache` pyramid:
1. `_render_glyph(char, color)` → single-character RGBA sprite (128 slots)
2. `_render_text(text, color)` → composites glyphs horizontally (512 slots)
3. `render_*_panel(...)` → composites text lines into panels (32–128 slots)

**Benefit**: Cleaner code, no module-level side effects. Robot/ball panels now benefit from per-line text caching — when only one robot moves, unchanged lines hit `_render_text` cache.

**Result**: hud_robot_r dropped from 0.2ms → 0.0ms. hud_step_r went from 0.0ms → 0.1ms (slight regression from glyph-per-char composition vs old prefix sprite, acceptable for code clarity).

### Pillow encode_to_file experiment

**Task**: Test whether Pillow's internal `_getencoder(...).encode_to_file(fd, bufsize)` can bypass both `tobytes()` chunked Python loop and the threading queue, writing raw pixels from C directly to the ffmpeg pipe fd.

**Reference**: [Pillow #5049](https://github.com/python-pillow/Pillow/issues/5049) — `tobytes()` internally loops in Python with 64KB chunks. For our ~830KB frames (576×480×3) that's ~13 iterations. The C encoder can write the full image in one call via `encode_to_file(fd, bufsize)`.

**Current path**: `bytes(memoryview(np.asarray(frame)))` → `queue.put(raw)` → writer thread → `stdin.write(raw)`.

**Proposed path**: `encoder.encode_to_file(pipe_fd, full_bufsize)` — zero Python-level copies, no bytes allocation, no queue.

**Tradeoff**: Loses threading overlap between render and encode. At 1050 steps current qput shows 0.0–0.5ms variance (often blocking on pipe), so the threading benefit may be small.

**Result (9000 steps)**: encode_to_file total=12.22s vs baseline 11.95s — **2% slower**. Synchronous encoding kills the threading parallelism. CPU% dropped 295→280%. Reverted.

### PyAV threaded encoder (KEPT)

**Task**: Replace ffmpeg subprocess + pipe with PyAV (libx264 in-process). Since PyAV releases the GIL during `stream.encode()` ([PyAV #303](https://github.com/PyAV-Org/PyAV/issues/303)), x264 work runs in a background thread truly in parallel with Python rendering.

**Architecture**: `render_state()` → `queue.put(pil_img)` → writer thread calls `av.VideoFrame.from_image(pil_img)` + `stream.encode(vf)` (GIL released during x264). No bytes serialization, no pipe, no subprocess.

**Result (9000 steps)**:
- Total: 11.95s → **9.72s** (-18.7%)
- Encode: 7.27s → 4.64s (-36%)
- tobytes: 0.2ms → 0.0ms (eliminated)
- CPU%: 295% → 356% (more parallelism)
- maxresident: 126MB → 143MB (+13%)

**At 1050 steps**: warm runs 1.05–1.15s (variance), vs baseline ~1.17s.

### Files changed
- `render_hud.py`: Removed `_DIGIT_GLYPHS`, `_DIGIT_H`, `_PREFIX_IMG`, `_PREFIX_TEXT` module globals. Added `_render_glyph()`, `_render_text()`. Rewrote `render_step_panel`, `render_robot_panel`, `render_ball_panel` to use the cache pyramid.
- `cli.py`: Replaced ffmpeg subprocess `_Encoder` with PyAV-based threaded encoder. Removed `fcntl`, `subprocess`, pipe buffer sizing. Added `av` dependency.
- `pyproject.toml`: Added `av` to dependencies.

---

## 2026-03-16: Zero-copy VideoFrame creation (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Eliminate GIL-holding `av.VideoFrame.from_image()` (~220µs/frame copying 829KB) by switching to zero-copy `from_numpy_buffer()`.

**Investigation**: Explored all PyAV VideoFrame factory methods:
- `from_image()`: allocates + copies 829KB twice (tobytes + memcpy). Holds GIL entire time.
- `from_numpy_buffer()`: sets plane pointers into existing numpy array memory. Zero copy. ~1-2µs GIL.
- `from_ndarray()`: may copy depending on format/layout — previously tested at 312µs (slower).
- `from_bytes()`: copies from raw bytes. Not useful here.
- Multi-threaded encoding (`thread_count=4, thread_type=3`): tested and rejected — encode halves but GIL contention inflates render by same amount. Net wash.

**Change**: In `_Encoder._writer()`:
```python
# Before (copies 829KB, holds GIL ~220µs):
vf = av.VideoFrame.from_image(pil_img)

# After (zero-copy pointer setup, GIL ~1-2µs):
npy = np.asarray(pil_img)  # view into PIL's existing pixel buffer
vf = av.VideoFrame.from_numpy_buffer(npy, format="rgb24")
```

**Why it works**: PIL already stores pixels as a contiguous C array. `np.asarray()` creates a 96-byte header pointing at that buffer (no copy). `from_numpy_buffer()` sets libav plane pointers to the same memory and holds a refcount to keep it alive. Two pointer setups vs two 829KB memcpys.

**Result (1050 steps, 3 runs each)**:
- Before: 0.91 / 1.06 / 1.17s (median ~1.06s), encode ~0.44s
- After: **0.88 / 0.91 / 0.94s** (median ~0.91s), encode ~0.38s
- Encode improvement: ~14% (0.44→0.38s)
- Best run: **0.88s** (sub-1s target achieved)

**Also tested (rejected):** Multi-threaded x264 encoding (`thread_count=4, thread_type=3`). Encode time halved (0.44→0.18s) but GIL contention from 4 encoder threads inflated render time by the same amount (0.54→0.82s). Total: wash (~1.05s). At 576×480 the per-frame encode is so short that thread dispatch overhead + GIL pressure dominate.

### Files changed
- `cli.py`: `_writer()` now uses `np.asarray()` + `av.VideoFrame.from_numpy_buffer()` instead of `av.VideoFrame.from_image()`.

---

## 2026-03-16: Switch to libx264rgb ultrafast (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

**Task**: Find fastest codec/preset combo for 576×480 RGB frames.

**Investigation**: Benchmarked 15 codec/preset combinations both in isolation (500 frames) and in real pipeline (5×1050 steps each). Also tested Pillow-SIMD (rejected — latest 9.5.0 missing `ImageFont.load_default(size=)` from Pillow 10+, project abandoned since 2023).

**Encoder thread breakdown** (per frame, instrumented):
- `qget_wait`: 0.01ms (queue never starves)
- `asarray`: 0.14ms (PIL `__array_interface__` still copies 829KB — not truly zero-copy)
- `from_buf`: 0.01ms (pointer setup)
- `encode`: 0.31ms with libx264rgb ultrafast (was 0.69ms with libx264 veryfast)
- `mux`: 0.01ms

**Real pipeline results (1050 steps, 5 runs each, on AC power)**:

| Config | Median total | Encode thread ms/frame |
|---|---|---|
| libx264 veryfast yuv420p (old) | 0.90s | 0.69 |
| libx264 ultrafast yuv420p | 0.66s | 0.46 |
| libx264rgb veryfast rgb24 | 0.81s | 0.56 |
| **libx264rgb ultrafast rgb24** | **0.54s** | **0.31** |

**9000 steps**: 8.99s → **3.87s** (-57%)

**Why libx264rgb ultrafast wins**:
1. Skips RGB→YUV420p color space conversion (~0.15ms/frame saved)
2. Ultrafast preset does less compression work (~0.2ms/frame saved)
3. Encoder thread finishes faster → zero queue backpressure → render thread gets full GIL access → render itself speeds up (0.43→0.37 ms/frame)

### Files changed
- `cli.py`: Encoder switched from `libx264`/`yuv420p`/`veryfast` to `libx264rgb`/`rgb24`/`ultrafast`.

---

## Multi-threaded render pipeline on Python 3.14t (Copilot, multi-session)

**Agent**: GitHub Copilot (Claude Opus 4.6)

### Motivation
Single-threaded rendering hit the GIL wall at ~0.54s/1050 steps. Moved to Python 3.14.3 free-threaded (deadsnakes PPA `python3.14-nogil`, `sys._is_gil_enabled()` → False) to enable true parallel rendering.

### Architecture

```
Main thread:     sim step → state.snapshot() → render_q
4 render workers: render_state() → np.asarray → av.VideoFrame(rgb24) → reformat(yuv420p) → enc.feed()
Encoder thread:  reorder buffer → stream.encode(vf) → container.mux(packet)
```

**Key design choices**:

1. **`WorldState.snapshot()`** — deep-copies mutable per-frame data (balls, poses, held_balls), shares static geometry references. Makes snapshots safe to read from any thread.

2. **`@lru_cache` returns `np.ndarray`** — PIL Image objects are NOT thread-safe for concurrent reads. All cached render functions (`_render_background`, `_make_agent_sprite`, HUD panels) return numpy arrays. Each thread creates its own `PIL.Image.fromarray()` when it needs to draw.

3. **Per-thread fonts via `threading.local()`** — FreeType font objects corrupt under concurrent access (segfaults, "invalid outline" errors). Each thread creates its own font instances via `_get_fonts()` in render.py and `_get_font_md()` in render_hud.py. Eliminates lock contention vs the earlier `_font_lock` approach.

4. **YUV420p conversion in render workers** — Render workers produce `av.VideoFrame` in yuv420p format. Encoder thread only does `stream.encode()` + `container.mux()` (inherently sequential x264 state + container IO).

5. **Reorder buffer in encoder** — Workers finish frames out of order. Encoder maintains `dict[int, VideoFrame]` keyed by step number, emitting consecutive frames to x264 in strict order.

6. **Encoder reverted to `libx264`/`veryfast`/`yuv420p`** — libx264rgb was optimal for single-threaded (skip colorspace conversion), but with multi-threaded workers doing the conversion in parallel, the standard yuv420p pipeline is preferred.

### Thread-safety bugs encountered & fixed
| Symptom | Root cause | Fix |
|---|---|---|
| Segfault with 4 workers | PIL Image concurrent reads | Cache numpy arrays, `Image.fromarray()` per thread |
| "invalid outline" / "unsupported glyph" | FreeType not thread-safe | `threading.local()` per-thread font objects |

### Performance (1000 steps, Python 3.14.3 free-threaded, Ryzen 5 7640U)

```
Total: 2.18s (2.18 ms/frame)
Encoder (bottleneck): encode=2.11ms mux=0.013ms → 2.17ms/frame
Render workers (4T, ms/frame):
  bg_copy=0.39 agents=0.32 hud_step=0.29 asarray=0.40 from_buf=0.98 yuv=3.55
  (parallelized → effective throughput ~1.6ms/frame)
Sim: 0.17s | Snap: 0.01s | Drain: 0.02s
```

### Files changed
- `push_back/env/state.py`: Added `WorldState.snapshot()` method
- `push_back/env/render.py`: `threading.local()` fonts, `@lru_cache` returns ndarray, `render_state()` accepts `timings` dict
- `push_back/env/render_hud.py`: Own `threading.local()` font, all render functions return ndarray
- `push_back/cli.py`: 4-worker render pipeline, reorder-buffer encoder, yuv420p conversion in workers

---

## x264 FRAME threading + timing fix (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

### Changes
1. **Fixed timing to add up** — added `qput` (render_q.put backpressure) and `join` (worker join wait) to output. Previously ~90% of wall time was unaccounted.
2. **x264 FRAME threading** — `stream.thread_count=2`, `stream.thread_type=FRAME`. x264 spawns 2 internal threads for frame-level parallelism.

### Benchmark grid (6-core Ryzen 5 7640U, 1000 steps, best of 2)

| Config | total ms/f | encode ms | file size |
|---|---|---|---|
| R4 x264=1 (old) | 1.52 | 1.48 | 297KB |
| **R4 x264=2F** | **1.35** | 0.60 | 297KB |
| R4 x264=3F | 1.36 | 0.38 | 297KB |
| R4 x264=6F | 1.50 | 0.41 | 310KB |
| R3 x264=1 | 1.56 | 1.51 | 297KB |
| R2 x264=1 | 1.87 | 1.41 | 297KB |
| R1 x264=1 | 2.79 | 1.28 | 297KB |

R4+x264=2F wins: 8 total threads on 6 cores is the sweet spot. More x264 threads steal CPU from render workers (qget_wait rises). Fewer render workers always hurts — render throughput is the bigger lever.

### Files changed
- `push_back/cli.py`: Added qput/join timing, set `thread_count=2` + `ThreadType.FRAME`

---

## Chunked parallel encoding experiment (Copilot) — REJECTED

**Agent**: GitHub Copilot (Claude Opus 4.6)

### Motivation
The old pipeline (4 render workers → queue → 1 encoder thread) is bottlenecked by the single encoder thread. User proposed: "could each thread render a sequence of frames and encode them independently?"

### Architecture
**Phase 1 — Simulate all steps upfront**: Run the entire simulation first, store `WorldState.snapshot()` after each step into `list[WorldState]`. Very fast (~0.05s for 1050 steps).

**Phase 2 — Chunked parallel render + encode**: Split snapshots into N contiguous chunks. Each worker thread independently:
1. Opens its own `av.open()` container writing to a temp `.mp4`
2. Creates its own `libx264` stream (veryfast, yuv420p)
3. Iterates through its chunk: `render_state()` → `np.asarray()` → `from_numpy_buffer(rgb24)` → `reformat(yuv420p)` → `stream.encode()` → `container.mux()`
4. Flushes and closes

No queues, no reorder buffer, no shared encoder. Thread-safety via `threading.local()` fonts + `@lru_cache` returning ndarray.

**Phase 3 — ffmpeg concat**: `ffmpeg -f concat -safe 0 -i concat.txt -c copy output.mp4` (stream copy, no re-encode, ~60ms). Temp files cleaned with `shutil.rmtree`.

### Benchmark (1050 steps, 6-core Ryzen 5 7640U, best of 2–3 runs)

| Config | ms/frame | File size |
|---|---|---|
| W=2 | 2.42 | 469 KB |
| W=3 | 2.02 | 466 KB |
| W=4 | 1.85 | 508 KB |
| W=5 | 1.82 | 439 KB |
| W=6 | 1.75 | 458 KB |
| **W=8** | **1.60** | 498 KB |
| W=10 | 1.67 | 537 KB |
| W=12 | 1.81 | 575 KB |
| W=16 | 1.87 | 653 KB |

Adding x264 internal FRAME threading per worker (tc=2F, tc=3F) made things worse due to total thread contention on 6 cores.

### Why it lost to the old pipeline
Best chunked: **1.60 ms/frame** (W=8) vs old pipeline: **1.35 ms/frame** (R4+x264=2F).

The critical difference is **pipelining**. In the old architecture, while the encoder thread encodes frame N, render workers are already rendering frames N+1..N+4 — render and encode overlap in time. In chunked, each worker does render→encode **serially**: it can't start rendering frame N+1 until it finishes encoding frame N. Wall time = render_time + encode_time, vs old pipeline ≈ max(render_time, encode_time).

More workers (W=8+) partially compensate by giving each worker fewer frames, but on 6 cores the oversubscription causes contention. The chunked approach would likely only win on machines with many more cores (16+).

### Decision
**Rejected** — reverted to old pipeline (R4 render workers + 1 encoder thread, x264 `thread_count=2` FRAME). Simpler code isn't worth 18% slower.

## PerfAccum timing refactor (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

Replaced all manual `t = time.perf_counter(); ...; dt += time.perf_counter() - t` timing with `PerfAccum` context-manager utility.

- Created `push_back/env/perf.py` — `PerfAccum` class with `section()` context manager, `add()`, `seconds()`, `total()`, `report_ms()`, `reset()`
- `render.py`: `_timings` dict → `_perf = PerfAccum()`, all manual timing → `with tm.section("key"):`, collapsed hud render/paste pairs into single sections
- `render_encoder.py`: `_enc_timings` dict → `PerfAccum`, eliminated `t0/t1/t2/t3/t4` juggling, `feed()` simplified to just `self._q.put(frame)` (removed cross-module `_timings` coupling)
- `cli.py`: `sim_dt`/`render_dt`/`encode_dt`/`drain_dt` → `PerfAccum`, 30-line report → 3-line report using `report_ms()`

## P-mode palette pipeline & VAAPI investigation (Copilot)

**Agent**: GitHub Copilot (Claude Opus 4.6)

### P-mode integration
Benchmark showed P/yuv420p/ultrafast/pal8 won at 1.43ms/frame (20% faster than RGB leader). Integrated full P-mode palette rendering:

- `render.py`: Background + balls render in P-mode with fixed 15-color palette. Added `_PALETTE_COLORS` tuple, `FLAT_PALETTE` (768 ints), `_P_*` palette index constants. Agents and HUD use cached numpy paste instead of PIL draw.
- `render_encoder.py`: `Encoder._writer` ingests P-mode PIL images via pal8 `VideoFrame` with BGRA palette on `plane[1]`. Palette computed once and cached.
- `cli.py`: Preview uses `libx264/yuv420p` instead of `libx264rgb/rgb24`.

### P-mode cache regression & fix
Initial integration lost all text/sprite LRU caches → massive regression (4.05ms render vs 0.60ms before).

Rewrote `render_hud.py` with P-mode cache pyramid (all caches produce `(H,W) uint8` palette index arrays):
- `_render_glyph(char, color_idx)` → renders char as L-mode mask, thresholds to palette index
- `_render_text(text, color_idx)` → composes cached glyphs into numpy row
- `render_step_panel()`, `render_robot_panel()`, `render_ball_panel()` → cached numpy panels with black bg
- `make_agent_sprite(heading, color_idx, red_n, blue_n)` → returns `(sprite, bool_mask)` cached arrays

**Result**: 0.85ms/frame total (was ~5.7ms) — **4.8× speedup**. Render: 0.31ms (agents 0.14ms from 0.84ms, HUD ~0.04ms from ~3.0ms). Encoder: 0.84ms (x264 encode 0.67ms is now the bottleneck).

### Timing accounting
Found 0.05ms untracked in render (`Image.fromarray` + `putpalette` at end). Added `to_pil` section — all timings now account within 0.02ms.

### Encode format benchmark
Benchmarked all preview formats with real rendered frames:
- x264/ultrafast/yuv420p: 0.320ms, **796KB** (optimal)
- rawvideo/pal8: 0.186ms, 317MB (too large)
- FFV1/pal8: ERROR (doesn't support pal8)
- png/pal8, huffyuv, zlib, qtrle: various errors or much slower

### VAAPI / hardware encode investigation
- System: AMD Ryzen 7640U, RDNA3 GFX1103, VCN 4.0 — `vainfo` confirms H.264 encode via VA-API
- System `ffmpeg` has `h264_vaapi`
- **PyAV 17.0.0's bundled ffmpeg does NOT include `h264_vaapi`** — only h264_nvenc, h264_qsv, h264_amf, h264_v4l2m2m, h264, libopenh264
- `h264_amf` listed but errors on Linux (requires Windows AMF runtime; AMD Linux = VAAPI)
- `h264_v4l2m2m` errors (ARM SoC only)
- `zscale` doesn't support pal8, and swscale pal8→yuv420p is only ~0.05ms anyway

**Decision**: x264 ultrafast at ~0.32ms/frame accepted as the PyAV encode floor. VAAPI would require building PyAV from source against system ffmpeg or subprocess pipe — marginal gain for significant complexity at 576×480.

### rawvideo vs x264 ultrafast (preview tmp file, 1200 steps)
| Encoder | ms/frame (encode thread) | /tmp file size |
|---|---|---|
| rawvideo/pal8 (.nut) | ~0.55 | ~500 MB |
| libx264/ultrafast/yuv420p | ~0.60–0.65 | ~1.2 MB |

rawvideo saves ~0.1ms/frame on encode but produces a ~400× larger tmp file in RAM (/tmp is tmpfs). Not worth the RAM pressure for a marginal speed gain.
