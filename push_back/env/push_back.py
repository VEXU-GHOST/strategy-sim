"""PettingZoo ParallelEnv for the VEX Push-Back game.

4 robots on a 36×36 grid (4"/cell).  Each tick is 0.1 s; 600 ticks = 60 s.
Actions are discrete.  Each agent slot is backed by a BaseRobot that owns
its observation space, action space, and action→primitive resolver.
"""

from __future__ import annotations

import functools

import gymnasium
import numpy as np
from pettingzoo import ParallelEnv

from push_back.env.robots.base import BaseRobot
from push_back.env.robots.stand_still import StandStill
from push_back.env.render import render_state
from push_back.env.field import make_default_goals, make_collision_segments
from push_back.env.collision import compute_blocked_cells, resolve_moves
from push_back.env.state import (
    BallColor,
    GRID_SIZE,
    ROBOT_RADIUS,
    WorldState,
    Heading,
    Pose,
)

DT: float = 0.1  # seconds per tick
DEFAULT_MAX_STEPS: int = 600  # 60 seconds


class PushBackEnv(ParallelEnv):
    """4-agent VEX field environment implementing PettingZoo ParallelEnv.

    Each agent slot is backed by a ``BaseRobot``.  The robot's
    ``tick`` method picks an action, and ``resolve`` maps it to a
    simulation primitive (STAY/FORWARD/TURN_*).
    """

    metadata = {
        "name": "push_back_v0",
        "render_modes": ["rgb_array", "human"],
    }

    possible_agents: list[str] = ["red_0", "red_1", "blue_0", "blue_1"]

    def __init__(
        self,
        n_balls: int = 10,
        render_mode: str | None = "rgb_array",
        max_steps: int = DEFAULT_MAX_STEPS,
        robots: dict[str, BaseRobot] | None = None,
    ) -> None:
        self.n_balls = n_balls
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.state = WorldState()
        self.agents: list[str] = []

        default: dict[str, BaseRobot] = {
            name: StandStill() for name in self.possible_agents
        }
        if robots:
            default.update(robots)
        self._robots: dict[str, BaseRobot] = default

    # ---- spaces ----

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gymnasium.Space:
        return self._robots[agent].observation_space(self.n_balls)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gymnasium.Space:
        return self._robots[agent].action_space()

    # ---- core API ----

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Randomize ball positions and place agents in starting corners."""
        self.agents = list(self.possible_agents)
        self.rng = np.random.default_rng(seed)

        positions = self.rng.integers(2, GRID_SIZE - 2, size=(self.n_balls, 2))
        colors = self.rng.choice([BallColor.RED, BallColor.BLUE], size=self.n_balls)
        balls_on_field = np.column_stack([positions, colors]).astype(np.int32)

        goals = make_default_goals()
        segments = make_collision_segments(goals)
        blocked = compute_blocked_cells(segments, ROBOT_RADIUS)

        for r in self._robots.values():
            r.reset()
            r.set_blocked_cells(blocked)
        margin = 4  # grid cells from edge
        starts = [
            Pose(margin, margin, Heading.E),
            Pose(margin, GRID_SIZE - 1 - margin, Heading.E),
            Pose(GRID_SIZE - 1 - margin, GRID_SIZE - 1 - margin, Heading.W),
            Pose(GRID_SIZE - 1 - margin, margin, Heading.W),
        ]

        self.state = WorldState(
            balls_on_field=balls_on_field,
            agents=starts,
            goals=goals,
            collision_segments=segments,
            blocked_cells=blocked,
            score=(0, 0),
            timestep=0,
        )
        observations = {a: self._get_obs() for a in self.agents}
        infos: dict[str, dict] = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]) -> tuple[
        dict[str, dict],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Apply all agent actions simultaneously and advance one tick."""
        proposed: list[Pose] = []
        for i, name in enumerate(self.possible_agents):
            if name not in actions:
                proposed.append(
                    Pose(
                        self.state.agents[i].x,
                        self.state.agents[i].y,
                        self.state.agents[i].heading,
                    )
                )
                continue
            robot = self._robots[name]
            proposed.append(robot.apply(actions[name], self.state.agents[i]))

        self.state.agents = resolve_moves(
            self.state.agents,
            proposed,
            self.state.blocked_cells,
            ROBOT_RADIUS,
        )

        self.state.timestep += 1
        truncated = self.state.timestep >= self.max_steps

        observations = {a: self._get_obs() for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos: dict[str, dict] = {a: {} for a in self.agents}

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self, *, draw_grid: bool = False) -> np.ndarray | None:
        """Render current state. Returns HWC uint8 array or None."""
        img = render_state(self.state, draw_grid=draw_grid)
        if self.render_mode == "human":
            img.show()
            return None
        return np.array(img)

    # ---- internals ----

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build observation dict from current state."""
        agent_arr = np.array(
            [[p.x, p.y, p.heading] for p in self.state.agents], dtype=np.int32
        )
        return {
            "balls": self.state.balls_on_field.copy(),
            "agents": agent_arr,
            "score": np.array(self.state.score, dtype=np.int32),
        }
