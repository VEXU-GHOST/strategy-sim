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
