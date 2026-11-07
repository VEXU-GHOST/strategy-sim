# Field environment following the whiteboard

Requires Python 3.10+ and NumPy (`python -m pip install numpy`). Extract the ZIP
and keep the `vex_field_environment` package folder intact. Run your own script
from the folder containing it, or put that folder on Python's import path.
Nothing starts automatically on import. There is no renderer, trainer or RL
policy in this package.

## Where to work

| File | Contents corresponding to the whiteboard |
| --- | --- |
| `states.py` | `FieldState`, four `RobotState`s, `Pose(x,y,theta)`, tasks, goals, inventories, supplies, clock data and prototype points. |
| `movement.py` | `AllowanceMap` of goals and walls; Dijkstra routes; constant linear/angular rates; a pose per tick; collision checks or explicit failure. |
| `actions.py` | Move, turn, pick up, score, toggle to our color, match-load, park, drop and wait; travel first, operate claw second; read-only consequence/time estimates. |
| `observations.py` | V0/full field and V1/cone observations, with range, heading and goal occlusion. |
| `vex_field_env.py` inside the package | Small coordinator: dispatch tasks, move all robots, resolve interactions, advance one shared clock, then return state. |
| `field_config.json` | Geometry, objects, speeds, interaction times, sensors and optional interaction locations. |

The top-level `vex_field_env.py` is only a compatibility import for older scripts.
Internal imports use package-relative names, so `actions.py` does not collide
with Amy's separate `v1/actions.py`.

## Bot example: collect, travel and score

```python
from vex_field_environment import Action, GameEnv

env = GameEnv()

def finish(action):
    state, rewards, done, info = env.step({"A0": action})
    if info["results"].get("A0", {}).get("status") == "rejected":
        raise RuntimeError(info["results"]["A0"])
    while state["robots"]["A0"]["busy"] and not done:
        state, rewards, done, info = env.step()
    print(info["results"].get("A0"))
    return state

pin = min(
    (obj for obj in env.object_states() if obj["item"] == "blue_yellow"),
    key=lambda obj: sum((a - b) ** 2 for a, b in zip(obj["position"], env.position("A0"))),
)
finish(Action("COLLECT", target=pin["object_id"]))

# SCORE now performs its own travel, alignment and interaction. It does not
# teleport the robot, and the goal's center remains a collision obstacle.
action = Action("SCORE", target="goal_0", item="blue_yellow", face="blue")
print(env.estimate_action("A0", action))
state = finish(action)
print(state["scores"])  # [red_score, blue_score]

sensor_view = env.observe("A0", mode="cone")
views = env.observe_all(mode="cone")
search_world = env.clone()  # Independent poses, tasks, clocks, stock and stacks.
```

The old `from vex_field_env import Action, GameEnv` import also works through
the top-level compatibility module.

## Actions, movement and time

Submit a task only when a robot is idle. A task may take many ticks:

1. Plan a static route to a reachable interaction position.
2. Turn in place and drive each segment at configured maximum rates.
3. If needed, face the target and spend the configured interaction time.
4. Recheck stock/cargo/capacity and apply the effect once.

`step({"A0": task_a, "A1": task_b})` starts both robots and advances just one
shared tick. `step()` continues existing tasks, while idle robots remain still.
You may instead call `start_action` for each robot without advancing time, then
call `step()`. All four robots' poses update before any completed interaction,
and observations reflect the final world after those changes.

`info["results"][robot_id]["status"]` is `running`, `complete`, `failed`, or
`rejected`. Running has `success=None`; an attempt is not a completed pickup
until the status says `complete`. `info["events"]` records completions/failures.
The observation has `busy`, `phase`, `remaining_time`, `wheels_active` and
`claw_active`. Wheels act during movement/alignment, claw during relevant
interactions, and both are inactive when idle.

| Action | Fields |
| --- | --- |
| `MOVE` | `position=(x,y)`, optional final `heading` |
| `TURN` | relative `angle` in -180..180 degrees |
| `COLLECT` / `PICKUP` | loose `object_id` as `target` |
| `SCORE` | `goal_id` as `target`, `item`, optional visible `face` |
| `TOGGLE` | configured toggle ID; only sets it to the acting alliance |
| `MATCHLOAD` | configured matchloader ID and `item` |
| `PARK` | configured parking-zone ID |
| `DROP` | `item` when cargo contains more than one type |
| `WAIT` / `IDLE` | no target |

`BLOCK` is explicitly reserved for a later version.

World coordinates are centered feet, with +x right and +y up. Heading 0 is east
and degrees increase counterclockwise. Loose objects use a one-inch count grid;
robot poses can interpolate continuously between grid cells. `movement.py`
turns a route into a start pose followed by one pose per configured tick.
Rates are constant except the final partial increment of each turn/segment.
Dijkstra minimizes grid travel distance, not turning time. Clear straight
segments bypass the grid search. It does not model a particular drivetrain.

Goals and walls are static obstacles enlarged by robot radius. Collision checks
cover swept segments and simultaneous robot crossings/swaps. Routes do not
forecast other robots; interference fails the current task so a policy can
wait/replan. Equal-time interactions use sorted robot ID and revalidate supply
and stack slots. Objects are consumed at completion, not reserved at dispatch.

Every tick is 0.25 seconds by default. Interaction durations round up to whole
ticks, and estimates use that same convention. `estimate_action` does not move
robots or change inventory. Estimates assume no later interference or target
changes. At match end, partial motion is retained and unfinished interactions
are cancelled without consuming or awarding items. `clone()` preserves pending
tasks and returns an independent hypothetical environment.

## Observations

`observe(mode="full")` / `mode="v0"` returns the full world, including independent
array copies. `state()` returns that same snapshot format.

`observe("A0", mode="cone")` / `mode="v1"` returns its own robot state, visible
objects/goals/toggles/loaders and visible robots' poses/teams. It excludes the
full board, unseen robots, unseen dynamic objects and others' pending tasks or
cargo. Range and FOV are configurable; goal bases occlude the view. The public
clock and scoreboard remain visible. This is a geometric sensor approximation,
not a camera/noise simulator. A policy using these observations may eventually
need memory or a belief state. These V0/V1 labels come from the whiteboard and
do not mean automatic compatibility with Amy's `v1` action API.

## Optional locations: populate from agreed field measurements

The reference JSON supplies no toggle/parking/physical matchloader locations.
Their arrays therefore stay empty; the action handlers exist but reject unknown
targets. The earlier `layout.loaders` line segments are retained as markings;
they do not silently become infinite object sources or toggles.

The following entries illustrate the schema only. Positions, supply counts and
bounds below are examples, not official field measurements:

```json
{
  "toggles": [{"id": "handle0", "position": [-5.0, 1.0], "quadrant": 0}],
  "matchloaders": [{
    "id": "blue_loader", "position": [-5.0, -4.0], "team": "blue",
    "stock": {"blue_yellow": 10, "cup": 10}
  }],
  "parking_zones": [{"id": "midfield", "bounds": [-1.5, 1.5, -1.5, 1.5]}]
}
```

Add such entries under `layout` after confirming their values. Quadrants are
0=NW, 1=NE, 2=SW, 3=SE. Toggle ownership is 0=neutral, 1=red, 2=blue. A loader
without a team is shared. Parking requires the circular body to fit inside its
zone; it sets `parked=True` but does not invent a parking-point rule.

## Existing model assumptions and V1 boundary

The supplied `vid_inputs(1).json` still provides **7 goals, 6 loose pins and 24
cups**, not Amy's design's 9 goals. A0's start comes from the reference; the other
three starts are placeholders. This refactor changes responsibilities and task
execution, not those unverified coordinates. There are no preloads by default.

Five types are blue/yellow, red/yellow, red/blue, yellow/yellow Pins, and Cup.
Grey and clear Cups share one type. Each robot has one Pin slot and one Cup
slot. Goals alternate Pin/Cup and have a default total capacity of 10 items.
Each placed Pin selects one visible face: red/blue earns 5 points for that color,
yellow earns 10 for its quadrant's owner, Cups earn no points. Display color is
not goal ownership. Previous faces remain visible. Neutral and central yellow
faces currently earn zero. Autonomous bonuses, AWP, physical stack occlusion,
phase restrictions and midfield-majority scoring remain unimplemented. These
are prototype assumptions, not a claim of official field/rule fidelity.

Rewards are each robot's team-score change per tick, with no learning bonuses.
Two allies share that reward; do not sum it as two independent game scores.

Raw fields `_board`, `_board_size`, `_agents`, `goalStates`, `goalLen`,
`fieldObject`, `zoneColors`, `redMatchload`, `blueMatchload` and
`calcGoalScore()` still expose what Amy's original bridge reads. Board layers
0..4 follow the type order above; goal slots use 0=empty and channel+1 otherwise.
The board contains only loose items, not robot/goal occupancy. Team A is blue
(score index 1) to match that original V1 convention. Its SCORE/TOGGLE bridge
handlers and masks still require wiring. The newer 55-action PPO interface
also needs an adapter.

Put this package inside `rl_game` to import it alongside `v1`. Use either the
original V1 wrapper's executor/clock or this environment's scheduler for a
rollout; do not mix them. The old executor bypasses native motion and still
uses its own 20-cells/second travel assumption and 60-second episode.

## Verification for this revision

Python/JSON syntax and short manual smoke checks were run for collection and
scoring, duration estimates, a Dijkstra detour, concurrent resource contention,
crossing-robot collisions, independent clones, cone observations, configured
toggle/matchload/parking interactions, and match-end cancellation. No new test
suite, training run, or integration run against Amy's policy was added.
