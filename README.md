# Push Back Strategy Simulator

A PettingZoo-based multi-agent simulation of the VEX Robotics "Push Back" game. Four robots (two red, two blue) operate on a 144×144 inch field with scattered balls. The environment exposes the standard PettingZoo `ParallelEnv` API so it can plug directly into multi-agent RL libraries.

Right now the project has the core simulation loop, a Pillow renderer that produces top-down MP4 output, and a couple of placeholder agents (random movement and stand-still). Scoring, game-over conditions, and actual strategy (MCTS / RL) are not yet implemented.

## Quickstart

```bash
# clone & create a virtual environment
git clone <repo-url> && cd strategy-sim
python3 -m venv .venv
source .venv/bin/activate

# install in editable mode + dev tooling (black, pre-commit hook)
pip install -e .
bash scripts/setup.sh

# minimal run (600 steps, output/output.mp4)
push-back

# common usage
push-back --steps 1200 --out outputs/demo --play
```

## Project Structure

```
push_back/
    __init__.py
    cli.py                  # typer CLI + PyAV threaded encoder
    env/
        __init__.py
        push_back.py        # PushBackEnv(ParallelEnv)
        state.py            # WorldState, Goal, BallColor, constants
        field.py            # make_default_goals(), field dimensions
        render.py           # Pillow renderer
        render_hud.py       # LRU-cached HUD panels
        planner.py          # greedy grid pathfinder
        collision.py        # segment-based collision geometry
        robots/
            __init__.py
            base.py         # BaseRobot ABC, BaseObservation
            stand_still.py  # StandStill (no-op robot)
            random_robot.py # RandomRobot (4-action kinematics)
            high_level.py   # GoToRobot (options / macro-actions)
            sweeper.py      # SweeperRobot (autonomous ball collection)
scripts/
    setup.sh                # installs dev deps + pre-commit hook
    compress.sh             # re-encode for smaller file size (Discord sharing)
```

## Viewing Output

`--play` opens mpv automatically after rendering. For manual playback:

```bash
mpv --loop outputs/demo.mp4 --pause --window-scale=2
```

Useful keybinds (at 10 fps):

| Key | Action |
|-----|--------|
| `.` / `,` | Step forward / backward one frame |
| `Shift+RIGHT` / `Shift+LEFT` | Seek ±1s (10 frames) exactly |
| `RIGHT` / `LEFT` | Seek ±5s (50 frames) |
| `]` / `[` | Speed up (2×) / slow down (0.5×) |
| `Space` | Pause / play |

## Understanding CLI Timing Output

A typical run produces output like:

```
encoder thread (ms/frame): qget=0.02 asarray=0.151 from_buf=0.018 encode=0.43 mux=0.010 | total=0.63 flush=0.7ms(total)
sim: 0.07s (0.07 ms/step) | render: 0.54s (0.52 ms/frame) | encode: 0.01s (0.01 ms/frame) | drain: 0.00s | total: 0.63s (0.59 ms/frame)
  render breakdown (ms/frame): bg_hash=0.01 bg_render=0.01 bg_copy=0.21 balls=0.03 agents=0.11 ...
  feed breakdown (ms/frame): tobytes=0.00 qput=0.00
```

### Top-level timings

Each frame goes through: **sim → render → encode** (piped to PyAV via threaded writer).

| Timer | What it measures |
|-------|------------------|
| **sim** | Physics step: build observations, call each robot's `tick()`, `env.step()`. |
| **render** | Pillow drawing: copy cached background, draw balls, paste agent sprites, paste HUD panels. |
| **encode** | Time in `enc.feed()` — enqueues PIL image for the writer thread. Blocks only if the 30-frame queue is full. |
| **drain** | Time in `enc.finish()` flushing remaining queued frames. Subset of encode. |
| **total** | Wall-clock for the full loop. |

**Parallelism**: A background thread encodes frames via PyAV (libx264rgb). While Python renders frame N+1, the encoder thread converts frame N to x264. The queue (depth 30) absorbs burst mismatches.

```
Main thread:    [sim][render][enqueue][sim][render][enqueue]...[drain]
Encoder thread:      [asarray+encode N-1][asarray+encode N]...
```

### Render breakdown

| Phase | What it draws |
|-------|---------------|
| **bg_hash** | Build hashable keys (frozenset/tuples) for LRU cache lookup. |
| **bg_render** | LRU-cached static background (grid, border, goals, axis labels). |
| **bg_copy** | `Image.copy()` — full-frame memcpy of the cached background. |
| **balls** | Aggregate balls by cell, draw colored circles/pie slices + count text. |
| **agents** | Look up LRU-cached robot sprites, `img.paste()` at grid positions. |
| **hud_step_r/p** | Render / paste the step-counter panel. |
| **hud_robot_r/p** | Render / paste the robot-positions panel. |
| **hud_ball_r/p** | Render / paste the ball-list panel. |

### Encoder thread breakdown

| Phase | What it does |
|-------|---------------|
| **qget** | Wait for next frame from queue. |
| **asarray** | `np.asarray(pil_img)` — PIL→numpy (internal copy via `__array_interface__`). |
| **from_buf** | `av.VideoFrame.from_numpy_buffer()` — pointer setup, ~zero cost. |
| **encode** | x264 encode (libx264rgb ultrafast, releases GIL). |
| **mux** | Write encoded packet to MP4 container. |
