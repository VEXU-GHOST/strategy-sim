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

# run the CLI (outputs MP4 by default, requires ffmpeg)
push-back
push-back --steps 120 --out outputs/demo
```

## Project Structure

```
push_back/
    __init__.py
    cli.py                  # typer CLI
    env/
        __init__.py
        push_back.py        # PushBackEnv(ParallelEnv)
        state.py            # WorldState, Goal, BallColor, constants
        field.py            # make_default_goals(), field dimensions
        render.py           # Pillow renderer
        planner.py          # greedy grid pathfinder
        robots/
            __init__.py
            base.py         # BaseRobot ABC, BaseObservation
            stand_still.py  # StandStill (no-op robot)
            random_robot.py # RandomRobot (4-action kinematics)
            high_level.py   # GoToRobot (options / macro-actions)
scripts/
    setup.sh                # installs dev deps + pre-commit hook
```

## Viewing Output

[mpv](https://mpv.io/) is the easiest way to inspect simulation output frame-by-frame.

```bash
# play the output
mpv --loop outputs/demo.mp4 --pause

# start at a specific time (e.g. 5s = step 50 at 10 fps)
mpv --loop outputs/demo.mp4 --pause --start=5
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
sim: 0.16s (0.2 ms/step) | render: 1.06s (1.2 ms/frame) | encode: 0.93s (1.0 ms/frame) | drain: 0.12s | total: 2.16s
  render breakdown (ms/frame): bg_hash=0.0 bg_copy=0.2 balls=0.1 agents=0.1 hud_step=0.4 hud_robot=0.3 hud_ball=0.1 tobytes=0.3
saved 901 frames → outputs/demo.mp4
```

### Top-level timings

Each frame goes through: **sim → render → encode** (piped to ffmpeg via threaded writer).

| Timer | What it measures |
|-------|------------------|
| **sim** | Physics step: build observations, call each robot's `tick()`, `env.step()`. Purely CPU, no I/O. |
| **render** | Pillow drawing: copy cached background, draw balls, paste agent sprites, paste HUD panels. Returns a PIL Image. |
| **encode** | Time spent on `enc.feed()` — converts frame to bytes and enqueues for the writer thread. Blocks only if the 30-frame queue is full (backpressure from ffmpeg). Includes drain time. |
| **drain** | Time spent in `enc.finish()` waiting for the writer thread to flush remaining queued frames and ffmpeg to exit. Subset of encode. |
| **total** | Wall-clock for the sim loop (sim + render + encode interleaved). |

**Parallelism**: A background thread writes frames to ffmpeg's stdin pipe (1 MB buffer). While Python renders frame N+1, the writer thread pushes frame N to ffmpeg which encodes concurrently. The queue (depth 30) absorbs burst mismatches.

```
Python:    [sim][render][enqueue][sim][render][enqueue]...[drain]
Writer:         [pipe write N-1 ][pipe write N        ]...
ffmpeg:         [encode N-1     ][encode N            ]...
```

### Render breakdown

| Phase | What it draws |
|-------|---------------|
| **bg_hash** | Build hashable keys (frozenset/tuples) from state for LRU cache lookup. |
| **bg_copy** | `Image.copy()` of the cached static background. Single largest cost — a full-frame memcpy each frame. |
| **balls** | Aggregate balls by cell, draw colored circles/pie slices + count text. |
| **agents** | Look up LRU-cached robot sprites, `img.paste()` at grid positions. |
| **hud_step** | Paste the step-counter panel (LRU-cached by step number). |
| **hud_robot** | Paste the robot-positions panel (LRU-cached by position tuple). |
| **hud_ball** | Paste the ball-list panel (LRU-cached by ball state tuple). |
| **tobytes** | `bytes(memoryview(np.asarray(frame)))` — frame-to-bytes conversion for the ffmpeg pipe. |
