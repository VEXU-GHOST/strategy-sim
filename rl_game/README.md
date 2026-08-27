# RL Game — 4-Agent Team Training Framework

A Python reinforcement learning framework for a **64×64 grid game** with **4 agents** (2 per team), built around DQN with experience replay and a target network.

---

## Project Structure

```
rl_game/
├── main.py                  # CLI entry point
├── config.py                # All hyperparameters & game settings
├── requirements.txt
├── env/
│   └── game_env.py          # 64×64 GameEnv — plug your rules in here
├── agents/
│   ├── base_agent.py        # Abstract interface
│   └── dqn_agent.py         # DQN agent (shared per team)
├── models/
│   └── dqn_net.py           # Convolutional Q-network
├── training/
│   ├── replay_buffer.py     # Experience replay
│   └── trainer.py           # Episode loop & logging
└── checkpoints/             # Auto-created on first save
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU support**: If you have a CUDA GPU, install the CUDA-enabled PyTorch wheel from https://pytorch.org/get-started/locally/ — the framework will use it automatically.

---

## Usage

### Start a new training run
```bash
python main.py train
```

### Custom episodes / steps
```bash
python main.py train --episodes 1000 --steps 200
```

### Resume from a checkpoint
```bash
python main.py train --resume checkpoints/team_a_final.pth checkpoints/team_b_final.pth
```

### Debug / watch the board render
```bash
python main.py train --episodes 5 --render
```

---

## Customising the Game

### 1. Adding new actions
Open `config.py` and add entries to the `Action` enum:
```python
class Action(IntEnum):
    UP     = 0
    DOWN   = 1
    LEFT   = 2
    RIGHT  = 3
    STAY   = 4
    ACTION = 5
    ATTACK = 6   # ← add here
```
`NUM_ACTIONS` updates automatically.

### 2. Implementing game rules
Edit `env/game_env.py` — three stubs await your logic:

| Method | Purpose |
|---|---|
| `_apply_actions()` | Move agents, handle collision, triggers |
| `_compute_rewards()` | Return per-agent reward signal |
| `_check_done()` | Define win/loss/draw conditions |

### 3. Tuning hyperparameters
All knobs are in `config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `EPISODES` | 500 | Total training episodes |
| `STEPS_PER_EPISODE` | 200 | Episode length cap |
| `GAMMA` | 0.99 | Future reward discount |
| `EPSILON_DECAY` | 0.995 | Exploration decay speed |
| `TRAIN_EVERY` | 10 | Steps between gradient updates |
| `BATCH_SIZE` | 64 | Replay sample size |
| `LR` | 1e-3 | Adam learning rate |

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                   Training Loop                     │
│                                                     │
│  env.reset() ──► state                              │
│       │                                             │
│       ▼                                             │
│  ┌─────────────────────────────────────────────┐   │
│  │  For each step:                             │   │
│  │   state ──► agent_a.act() ──► A0, A1 acts  │   │
│  │   state ──► agent_b.act() ──► B0, B1 acts  │   │
│  │                  │                          │   │
│  │           env.step(actions)                 │   │
│  │                  │                          │   │
│  │   next_state, rewards, done                 │   │
│  │                  │                          │   │
│  │   agent_a.remember(...)  ←  A0, A1 rewards  │   │
│  │   agent_b.remember(...)  ←  B0, B1 rewards  │   │
│  │                  │                          │   │
│  │   every TRAIN_EVERY steps:                  │   │
│  │     agent_a.train_step()  (gradient update) │   │
│  │     agent_b.train_step()  (gradient update) │   │
│  └─────────────────────────────────────────────┘   │
│       │                                             │
│  Decay ε, sync target nets, save checkpoints        │
└─────────────────────────────────────────────────────┘
```
