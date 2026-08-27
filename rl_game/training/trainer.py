"""
training/trainer.py — Orchestrates episode rollouts and model training.

The Trainer owns:
  - The GameEnv instance
  - Two DQNAgent instances (one per team, shared across teammates)
  - The main training loop

Flow per episode:
  1. env.reset() → initial state
  2. For each step:
       a. Broadcast state to all 4 agents → collect actions
       b. env.step(actions) → next_state, rewards, done
       c. Store transitions in each team's replay buffer
       d. Every TRAIN_EVERY steps → trigger train_step() for both teams
  3. Decay epsilon
  4. Every TARGET_UPDATE_EVERY episodes → sync target networks
  5. Every CHECKPOINT_EVERY episodes → save checkpoints
"""

from __future__ import annotations

import os
import time
from functools import cmp_to_key
from typing import Dict, Optional

import numpy as np
import sys
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TEAM_A,
    TEAM_B,
    TRAIN_EVERY,
    TARGET_UPDATE_EVERY,
    CHECKPOINT_EVERY,
    CHECKPOINT_DIR,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    LOG_EVERY,
    EPISODES,
)
from env.game_env import GameEnv
from agents.dqn_agent import DQNAgent


class Trainer:
    """
    Manages two team agents and drives the environment loop.

    Args:
        episodes         : number of episodes to train for
        render           : if True, call env.render() every step
        resume_team_a    : path to Team-A checkpoint (optional)
        resume_team_b    : path to Team-B checkpoint (optional)
    """

    def __init__(
        self,
        episodes: int = EPISODES,
        render: bool = False,
        resume_team_a: Optional[str] = None,
        resume_team_b: Optional[str] = None,
    ):
        self.episodes = episodes
        self.render = render

        self.env = GameEnv()

        # One DQNAgent per team — both teammates share the same model weights
        self.agent_a = DQNAgent(team=TEAM_A, agent_id="TeamA")
        self.agent_b = DQNAgent(team=TEAM_B, agent_id="TeamB")

        if resume_team_a:
            self.agent_a.load(resume_team_a)
        if resume_team_b:
            self.agent_b.load(resume_team_b)

        self.epsilon = EPSILON_START
        self._global_step = 0

    def _build_comb_arr(self, state: dict, focus_agent: str) -> dict:
        """Build the `combArr` structure for a specific agent's perspective.

        Args:
            state: environment state dict (as produced by `env.reset()` / `env.step()`)
            focus_agent: agent id string, one of "A0","A1","B0","B1"

        Returns:
            combArr dict (same shape used when calling `agent.act()`)
        """
        baseArr = copy.deepcopy(state["board"])
        Arr2 = [[0] * 7 for _ in range(7)]
        
        for i in range(len(baseArr)):
            for j in range(len(baseArr[i])):
                if baseArr[i][j][0] == 200:
                    continue
                Arr2[baseArr[i][j][0] // 5][baseArr[i][j][1] // 5] += 1

        focus_row = state["agent_positions"][focus_agent][0]
        focus_col = state["agent_positions"][focus_agent][1]
        Arr1 = baseArr
        for i in range(len(Arr1)):
            for j in range(len(Arr1[i])):
                Arr1[i][j][0] -= focus_row
                Arr1[i][j][1] -= focus_col

        # Sort each row by squared distance from the focus agent
        Arr1 = [
            sorted(row, key=lambda p: (p[0] * p[0] + p[1] * p[1]))
            for row in Arr1
        ]

        ap = state["agent_positions"]
        # Map team-mate ordering so that the focused agent appears under the
        # expected key (this mirrors existing code which swaps A0/A1 or B0/B1
        # for an agent's own perspective).
        agent_positions = {
            "A0": ap["A0"],
            "A1": ap["A1"],
            "B0": ap["B0"],
            "B1": ap["B1"],
        }
        if focus_agent in ("A0", "A1"):
            other = "A1" if focus_agent == "A0" else "A0"
            agent_positions["A0"] = ap[focus_agent]
            agent_positions["A1"] = ap[other]
        else:
            other = "B1" if focus_agent == "B0" else "B0"
            agent_positions["B0"] = ap[focus_agent]
            agent_positions["B1"] = ap[other]

        combArr = {
            "board": [Arr1, Arr2.copy()],
            "agent_positions": agent_positions,
            "goalStates": state["goalStates"],
            "loaderStates": state["loaderStates"],
            "zoneColors": state["zoneColors"],
        }

        return combArr

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the training loop."""
        print(
            f"\n{'-'*60}\n"
            f"  RL Training - {self.episodes} episodes\n"
            f"  Device: {self.agent_a.device}\n"
            f"{'-'*60}\n"
        )

        for episode in range(1, self.episodes + 1):
            ep_start = time.time()
            stats = self._run_episode()
            elapsed = time.time() - ep_start

            # --- Epsilon decay ---
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

            # --- Target network sync ---
            if episode % TARGET_UPDATE_EVERY == 0:
                self.agent_a.sync_target_net()
                self.agent_b.sync_target_net()

            # --- Checkpointing ---
            if episode % CHECKPOINT_EVERY == 0:
                self._save_checkpoints(episode)

            # --- Logging ---
            if episode % LOG_EVERY == 0:
                self._log(episode, stats, elapsed)

        print("\n[Trainer] Training complete.")
        self._save_checkpoints("final")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_episode(self) -> Dict:
        """
        Run a single episode from reset to done.
        Returns a stats dict for logging.
        """
        state = self.env.reset()
        total_rewards = {"A0": 0.0, "A1": 0.0, "B0": 0.0, "B1": 0.0}
        losses_a, losses_b = [], []
        step = 0
        userInput = False

        def compare_by_distance(a, b) -> int:
            dist_a = a[0] * a[0] + a[1] * a[1]
            dist_b = b[0] * b[0] + b[1] * b[1]
            if dist_a < dist_b:
                return -1
            if dist_a > dist_b:
                return 1
            return 0

        while True:
            step += 1
            self._global_step += 1
            combArr = self._build_comb_arr(state, "A0")
            move1 = self.agent_a.act(combArr, userInput, self.epsilon)

            combArr = self._build_comb_arr(state, "A1")
            move2 = self.agent_a.act(combArr, userInput, self.epsilon)


            combArr = self._build_comb_arr(state, "B0")
            move3 = self.agent_b.act(combArr, userInput, self.epsilon)


            combArr = self._build_comb_arr(state, "B1")
            move4 = self.agent_b.act(combArr, userInput, self.epsilon)
            # ── 1. Each agent chooses an action ───────────────────────
            # Both Team-A agents share agent_a's model; same for Team-B.
            actions = {
                "A0": move1,
                "A1": move2,
                "B0": move3,
                "B1": move4
            }

            # ── 2. Step the environment ────────────────────────────────
            next_state, rewards, done, _info = self.env.step(actions)

            if self.render:
                self.env.render()

            # ── 3. Store transitions ───────────────────────────────────

            aid = "A0"
            combArr = self._build_comb_arr(state, aid)
            combArr1 = self._build_comb_arr(next_state, aid)
            self.agent_a.remember(combArr, actions[aid], rewards[aid], combArr1, done)
            total_rewards[aid] += rewards[aid]      

            aid = "A1"
            combArr = self._build_comb_arr(state, aid)
            combArr1 = self._build_comb_arr(next_state, aid)
            self.agent_a.remember(combArr, actions[aid], rewards[aid], combArr1, done)
            total_rewards[aid] += rewards[aid]  

            aid = "B0"
            combArr = self._build_comb_arr(state, aid)
            combArr1 = self._build_comb_arr(next_state, aid)
            self.agent_a.remember(combArr, actions[aid], rewards[aid], combArr1, done)
            total_rewards[aid] += rewards[aid]    

            aid = "B1"
            combArr = self._build_comb_arr(state, aid)
            combArr1 = self._build_comb_arr(next_state, aid)
            self.agent_a.remember(combArr, actions[aid], rewards[aid], combArr1, done)
            total_rewards[aid] += rewards[aid]              
            

            state = next_state

            # ── 4. Train every TRAIN_EVERY steps ──────────────────────
            if self._global_step % TRAIN_EVERY == 0:
                loss_a = self.agent_a.train_step()
                loss_b = self.agent_b.train_step()
                if loss_a is not None:
                    losses_a.append(loss_a)
                if loss_b is not None:
                    losses_b.append(loss_b)

            if done:
                break

        return {
            "steps": step,
            "reward_a": sum(total_rewards[k] for k in ("A0", "A1")),
            "reward_b": sum(total_rewards[k] for k in ("B0", "B1")),
            "loss_a": (sum(losses_a) / len(losses_a)) if losses_a else None,
            "loss_b": (sum(losses_b) / len(losses_b)) if losses_b else None,
        }
    def calcDist(x1, y1, x2, y2)-> int:
        distx = x1-x2
        disty = y1-y2
        return distx*distx+disty*disty

    def _save_checkpoints(self, tag) -> None:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.agent_a.save(os.path.join(CHECKPOINT_DIR, f"team_a_{tag}.pth"))
        self.agent_b.save(os.path.join(CHECKPOINT_DIR, f"team_b_{tag}.pth"))

    def _log(self, episode: int, stats: Dict, elapsed: float) -> None:
        loss_a = f"{stats['loss_a']:.4f}" if stats["loss_a"] is not None else "  N/A  "
        loss_b = f"{stats['loss_b']:.4f}" if stats["loss_b"] is not None else "  N/A  "
        print(
            f"[Ep {episode:>5}/{self.episodes}] "
            f"steps={stats['steps']:>4}  "
            f"rew_A={stats['reward_a']:>7.2f}  "
            f"rew_B={stats['reward_b']:>7.2f}  "
            f"loss_A={loss_a}  loss_B={loss_b}  "
            f"eps={self.epsilon:.3f}  "
            f"({elapsed:.1f}s)"
        )
