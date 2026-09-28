"""A small Markov Decision Process and MCTS testbed.

This module deliberately has no VEX, Manim, NumPy, or PyTorch dependency.
It is a safe place to test the MDP ideas before adapting the same interface to
the full GameEnv.

NOTICE: THIS RUNS INDEPENDENTLY OF EVERYTHING ELSE! FOR REFERENCE ONLY!

Run:
    python mdp_mcts_sandbox.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import random
from typing import Dict, Iterable


class Action(Enum):
    """Actions are distinct from states, as in the MDP notes."""

    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    PICK_UP_PIN = auto()
    SCORE_PIN = auto()


MOVE_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class FieldState:
    """Everything needed to predict the next step.

    This is the Markov property in code: the transition never needs the full
    earlier history.  Later, replace this toy state with a compact VEX state,
    such as robot position, held item, goal status, score, and time remaining.
    """

    row: int
    col: int
    carrying_pin: bool
    score: int
    steps_left: int


class TinyFieldMDP:
    """A stochastic field: collect one pin and bring it to the scoring goal."""

    gamma = 0.95

    def __init__(self) -> None:
        self.rows = 5
        self.cols = 5
        self.start = FieldState(0, 0, False, 0, 18)
        self.pin_location = (2, 1)
        self.goal_location = (4, 4)
        self.walls = {(1, 1), (1, 2), (3, 2)}

    def legal_actions(self, state: FieldState) -> list[Action]:
        """Return A(s), the actions currently available in this state."""
        if self.is_terminal(state):
            return []

        actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        if (state.row, state.col) == self.pin_location and not state.carrying_pin:
            actions.append(Action.PICK_UP_PIN)
        if (state.row, state.col) == self.goal_location and state.carrying_pin:
            actions.append(Action.SCORE_PIN)
        return actions

    def is_terminal(self, state: FieldState) -> bool:
        return state.score >= 1 or state.steps_left <= 0

    def transition_probabilities(
        self,
        state: FieldState,
        action: Action,
    ) -> Dict[Action, float]:
        """Return P(actual movement | state, requested action).

        A requested movement works 80% of the time and slips to either
        perpendicular direction 10% of the time.  Non-movement actions are
        deterministic.  This is the explicit P part of an MDP.
        """
        if action not in MOVE_DELTAS:
            return {action: 1.0}

        slips = {
            Action.UP: (Action.LEFT, Action.RIGHT),
            Action.DOWN: (Action.RIGHT, Action.LEFT),
            Action.LEFT: (Action.DOWN, Action.UP),
            Action.RIGHT: (Action.UP, Action.DOWN),
        }
        left_slip, right_slip = slips[action]
        return {action: 0.8, left_slip: 0.1, right_slip: 0.1}

    def step(
        self,
        state: FieldState,
        action: Action,
        rng: random.Random,
    ) -> tuple[FieldState, float, bool]:
        """Sample S', R, and terminal status from S and A."""
        if action not in self.legal_actions(state):
            raise ValueError(f"{action.name} is not legal in {state}")

        if action in MOVE_DELTAS:
            actual_action = self._sample_action(self.transition_probabilities(state, action), rng)
            row_change, col_change = MOVE_DELTAS[actual_action]
            new_row = min(max(state.row + row_change, 0), self.rows - 1)
            new_col = min(max(state.col + col_change, 0), self.cols - 1)

            # A wall leaves the robot where it is and costs more than a normal step.
            hit_wall = (new_row, new_col) in self.walls
            if hit_wall:
                new_row, new_col = state.row, state.col
            reward = -0.30 if hit_wall else -0.05
            next_state = FieldState(
                new_row,
                new_col,
                state.carrying_pin,
                state.score,
                state.steps_left - 1,
            )
        elif action == Action.PICK_UP_PIN:
            next_state = FieldState(
                state.row,
                state.col,
                True,
                state.score,
                state.steps_left - 1,
            )
            reward = 0.50
        else:  # Action.SCORE_PIN
            next_state = FieldState(
                state.row,
                state.col,
                False,
                state.score + 1,
                state.steps_left - 1,
            )
            reward = 10.0

        return next_state, reward, self.is_terminal(next_state)

    @staticmethod
    def _sample_action(probabilities: Dict[Action, float], rng: random.Random) -> Action:
        threshold = rng.random()
        cumulative = 0.0
        for action, probability in probabilities.items():
            cumulative += probability
            if threshold <= cumulative:
                return action
        return next(reversed(probabilities))  # Protect against tiny float-rounding errors.


@dataclass
class MCTSNode:
    """A node represents one state; its incoming edge represents one action."""

    state: FieldState
    parent: MCTSNode | None = None
    action_from_parent: Action | None = None
    reward_from_parent: float = 0.0
    children: dict[Action, MCTSNode] = field(default_factory=dict)
    untried_actions: list[Action] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0

    @property
    def value(self) -> float:
        """V(s): the rollout estimate of expected future discounted reward."""
        return self.total_value / self.visits if self.visits else 0.0


class MCTSPlanner:
    """UCT Monte Carlo Tree Search over an MDP simulator."""

    def __init__(
        self,
        mdp: TinyFieldMDP,
        iterations: int = 400,
        exploration_constant: float = math.sqrt(2),
        rollout_depth: int = 20,
        seed: int = 7,
    ) -> None:
        self.mdp = mdp
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    def choose_action(self, state: FieldState) -> tuple[Action, MCTSNode]:
        """Search from a state and return the action with the best estimated Q(s, a)."""
        root = MCTSNode(state=state, untried_actions=self.mdp.legal_actions(state))

        for _ in range(self.iterations):
            node = self._select(root)

            if not self.mdp.is_terminal(node.state) and node.untried_actions:
                node = self._expand(node)

            rollout_value = self._rollout(node.state)
            self._backpropagate(node, rollout_value)

        if not root.children:
            raise RuntimeError("Cannot choose an action from a terminal state.")

        # Q(s, a) = immediate reward + gamma * V(s').
        action = max(root.children, key=lambda candidate: self._action_value(root.children[candidate]))
        return action, root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection: follow UCB until a node has an untried action."""
        while not self.mdp.is_terminal(node.state) and not node.untried_actions and node.children:
            node = max(node.children.values(), key=self._ucb_score)
        return node

    def _expand(self, parent: MCTSNode) -> MCTSNode:
        """Expansion: add one action, rather than expanding every possible child."""
        action = self.rng.choice(parent.untried_actions)
        parent.untried_actions.remove(action)
        next_state, reward, _ = self.mdp.step(parent.state, action, self.rng)
        child = MCTSNode(
            state=next_state,
            parent=parent,
            action_from_parent=action,
            reward_from_parent=reward,
            untried_actions=self.mdp.legal_actions(next_state),
        )
        parent.children[action] = child
        return child

    def _rollout(self, state: FieldState) -> float:
        """Simulation: estimate V(s) using a random policy and discounted rewards."""
        value = 0.0
        discount = 1.0
        current_state = state

        for _ in range(self.rollout_depth):
            if self.mdp.is_terminal(current_state):
                break
            action = self.rng.choice(self.mdp.legal_actions(current_state))
            current_state, reward, _ = self.mdp.step(current_state, action, self.rng)
            value += discount * reward
            discount *= self.mdp.gamma

        return value

    def _backpropagate(self, node: MCTSNode, child_value: float) -> None:
        """Backpropagation: carry V(s') and its reward back through parent nodes."""
        value = child_value
        while node is not None:
            node.visits += 1
            node.total_value += value
            value = node.reward_from_parent + self.mdp.gamma * value
            node = node.parent

    def _action_value(self, child: MCTSNode) -> float:
        return child.reward_from_parent + self.mdp.gamma * child.value

    def _ucb_score(self, child: MCTSNode) -> float:
        if child.visits == 0:
            return math.inf
        parent_visits = max(child.parent.visits, 1)
        exploitation = self._action_value(child)
        exploration = self.exploration_constant * math.sqrt(math.log(parent_visits) / child.visits)
        return exploitation + exploration


class TabularQLearner:
    """Optional TD learner: Q(s,a) <- Q(s,a) + alpha * TD error.

    MCTS does not require this learner.  It is included to connect the notes'
    temporal-difference equation to code and to give you a later RL baseline.
    """

    def __init__(self, mdp: TinyFieldMDP, alpha: float = 0.15) -> None:
        self.mdp = mdp
        self.alpha = alpha
        self.q_values: dict[tuple[FieldState, Action], float] = {}

    def q(self, state: FieldState, action: Action) -> float:
        return self.q_values.get((state, action), 0.0)

    def update(self, state: FieldState, action: Action, reward: float, next_state: FieldState, done: bool) -> float:
        """Return the TD error: R + gamma max_a' Q(S', a') - Q(S, A)."""
        future = 0.0 if done else max(
            (self.q(next_state, next_action) for next_action in self.mdp.legal_actions(next_state)),
            default=0.0,
        )
        td_error = reward + self.mdp.gamma * future - self.q(state, action)
        self.q_values[(state, action)] = self.q(state, action) + self.alpha * td_error
        return td_error


def render_state(mdp: TinyFieldMDP, state: FieldState) -> str:
    """A compact visual check for the independent MDP sandbox."""
    rows = []
    for row in range(mdp.rows):
        symbols = []
        for col in range(mdp.cols):
            point = (row, col)
            if point == (state.row, state.col):
                symbols.append("R")
            elif point in mdp.walls:
                symbols.append("#")
            elif point == mdp.pin_location and not state.carrying_pin:
                symbols.append("P")
            elif point == mdp.goal_location:
                symbols.append("G")
            else:
                symbols.append(".")
        rows.append(" ".join(symbols))
    return "\n".join(rows)


def demo() -> None:
    """Run an MCTS-controlled episode without the full VEX project."""
    mdp = TinyFieldMDP()
    # 1,000 rollouts is still quick for this tiny world, but reliably discovers
    # the collect-then-score route despite stochastic movement.
    planner = MCTSPlanner(mdp, iterations=1_000, seed=12)
    environment_rng = random.Random(99)
    state = mdp.start

    print("Legend: R = robot, P = pin, G = goal, # = wall")
    print(render_state(mdp, state))

    while not mdp.is_terminal(state):
        action, root = planner.choose_action(state)
        next_state, reward, done = mdp.step(state, action, environment_rng)
        print(
            f"\nstate={state} | action={action.name} | "
            f"estimated Q={planner._action_value(root.children[action]):.2f} | reward={reward:.2f}"
        )
        state = next_state
        print(render_state(mdp, state))
        if done:
            break

    print(f"\nFinished: score={state.score}, steps left={state.steps_left}")


if __name__ == "__main__":
    demo()
