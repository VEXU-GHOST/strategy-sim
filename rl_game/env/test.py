import math
from vex_field_environment import Action, GameEnv

print("1: test.py started", flush=True)

import math
from vex_field_environment import Action, GameEnv

print("2: imports finished", flush=True)

env = GameEnv()

print("3: environment created", flush=True)


def run_action(action):
    state, rewards, done, info = env.step({"A0": action})

    result = info["results"]["A0"]
    if result["status"] == "rejected":
        raise RuntimeError(result)

    # Continue advancing time until this action finishes.
    while state["robots"]["A0"]["busy"] and not done:
        state, rewards, done, info = env.step()

    result = info["results"]["A0"]
    if result["status"] != "complete":
        raise RuntimeError(result)

    print(f"{action.verb}: complete at {env.time:.2f} seconds")
    return state


# Find the closest blue/yellow pin.
pin = min(
    (obj for obj in env.object_states()
     if obj["item"] == "blue_yellow"),
    key=lambda obj: math.dist(env.position("A0"), obj["position"]),
)

# Pick it up.
state = run_action(Action("COLLECT", target=pin["object_id"]))
assert state["robots"]["A0"]["held_items"][0] == 1

# Travel to a goal and score it.
state = run_action(
    Action("SCORE", target="goal_0", item="blue_yellow", face="blue")
)

# Check that the pin transferred from the robot to the goal.
assert state["robots"]["A0"]["held_items"].sum() == 0
assert len(state["goals"]["goal_0"]["stack"]) == 1
assert state["scores"] == [0.0, 5.0]

print("Scores:", state["scores"])
print("Basic environment check passed.")