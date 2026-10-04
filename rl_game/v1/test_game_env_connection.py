"""
Optional connection test for the teammate's real GameEnv.

Run from repository root:
    python rl_game\\v1\\test_game_env_connection.py

This file intentionally does not modify GameEnv. It only checks whether
GameEnv imports and whether the V1 bridge can read its state.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RL_GAME_DIR = os.path.dirname(HERE)
if RL_GAME_DIR not in sys.path:
    sys.path.insert(0, RL_GAME_DIR)

from strategy_env import StrategyEnv


try:
    from env.game_env import GameEnv
except Exception as exc:
    print("GameEnv import failed before V1 could connect to it.")
    print("This means the blocker is inside the current full GameEnv, not the V1 layer.")
    raise


env = StrategyEnv(GameEnv())
state = env.reset()

print("GameEnv connection OK")
print("A0:", state["agents"]["A0"])
print("A1:", state["agents"]["A1"])
print("objects read from full GameEnv:", len(state["object_states"]))
print("A0 actions:", [str(a) for a in env.get_available_actions("A0")])
