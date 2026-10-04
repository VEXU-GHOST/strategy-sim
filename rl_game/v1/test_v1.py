import numpy as np

from actions import ActionType
from strategy_env import StrategyEnv


class FakeAgent:
    def __init__(self, row, col, team=0):
        self.row = row
        self.col = col
        self.team = team
        self.held_items = np.zeros(5, dtype=float)


class FakeGameEnv:
    """
    Minimal stand-in with the same fields that V1 reads from full GameEnv.
    This test checks the V1 architecture without depending on teammate code.
    """

    def __init__(self):
        self._board_size = 20
        self.reset()

    def reset(self):
        self._board = np.zeros((5, self._board_size, self._board_size), dtype=int)

        # Two different regions so the two agents get different durations.
        self._board[0, 2, 2] = 1      # q0
        self._board[1, 17, 17] = 1    # q3

        self._agents = {
            "A0": FakeAgent(1, 1, team=0),
            "A1": FakeAgent(1, 2, team=0),
            "B0": FakeAgent(18, 18, team=1),
            "B1": FakeAgent(18, 17, team=1),
        }

        self.goalStates = np.zeros((9, 10), dtype=int)
        self.zoneColors = np.zeros(4, dtype=int)
        self.redMatchload = np.zeros(5, dtype=int)
        self.blueMatchload = np.zeros(5, dtype=int)
        self.fieldObject = np.array(
            [
                [2, 5, 0],
                [5, 2, 0],
                [17, 14, 0],
                [14, 17, 0],
            ],
            dtype=int,
        )

        return {}

    def calcGoalScore(self):
        return [0, 0]


env = StrategyEnv(FakeGameEnv())
state = env.reset()

print("=== INITIAL V1 STATE ===")
print("time:", state["time"])
print("A0:", state["agents"]["A0"])
print("A1:", state["agents"]["A1"])
print("collect regions:", sorted({obj["region"] for obj in state["object_states"]}))


a0_actions = env.get_available_actions("A0")
a1_actions = env.get_available_actions("A1")

print("\nA0 available:", [str(a) for a in a0_actions])
print("A1 available:", [str(a) for a in a1_actions])


a0_collect = next(
    a for a in a0_actions
    if a.action_type == ActionType.COLLECT and a.target == "q0"
)

a1_collect = next(
    a for a in a1_actions
    if a.action_type == ActionType.COLLECT and a.target == "q3"
)


env.start_action("A0", a0_collect)
env.start_action("A1", a1_collect)

state, reward, done, info = env.advance_to_next_event()

print("\n=== FIRST EVENT ===")
print("time:", state["time"])
print("completed:", info["completed_agents"])
print("reward:", reward)
print("A0:", state["agents"]["A0"])
print("A1:", state["agents"]["A1"])

assert state["time"] > 0.0
assert reward == 0.0
assert "A0" in info["completed_agents"]
assert state["agents"]["A0"]["carrying"] is True
assert state["agents"]["A1"]["busy"] is True

print("\nV1 architecture test passed!")
