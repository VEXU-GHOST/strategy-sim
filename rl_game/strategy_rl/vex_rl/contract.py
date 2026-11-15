"""Stable interface to implement when the teammate's full simulator is ready."""
from __future__ import annotations
from typing import Protocol
import numpy as np


class CentralizedStrategyEnv(Protocol):
    """One friendly robot is selected for each decision; actions may run concurrently.

    Input action_id follows ActionCodec(config) order.
    step info MUST include delta_time (actual elapsed simulated seconds).
    The returned reward MUST use the same team-oriented reward specification.
    """
    active_robot: str | None
    encoder: object
    codec: object

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]: ...
    def action_mask(self, robot_id: str | None = None) -> np.ndarray: ...
    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    def state(self, team: int = 0, active_robot: str | None = None) -> dict: ...


def check_contract(env: CentralizedStrategyEnv) -> None:
    observation, info = env.reset(seed=77)
    mask = env.action_mask()
    if observation.shape != (env.encoder.obs_dim,):
        raise AssertionError("observation dimensions differ")
    if mask.shape != (len(env.codec),) or mask.dtype != np.bool_ or not mask.any():
        raise AssertionError("action mask dimensions/type invalid")
    if not np.array_equal(info["action_mask"], mask):
        raise AssertionError("reset info mask does not match action_mask()")
    next_obs, reward, done, truncated, meta = env.step(int(np.flatnonzero(mask)[0]))
    if next_obs.shape != observation.shape or not np.isfinite(next_obs).all():
        raise AssertionError("step returned invalid observation")
    if not np.isfinite(reward) or meta.get("delta_time", -1) < 0:
        raise AssertionError("step reward/delta_time invalid")
    if done and env.active_robot is not None:
        raise AssertionError("terminal env still declares an active robot")
