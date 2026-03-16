"""SweeperRobot — methodical boustrophedon sweep to collect own-color balls.

Navigates a lawn-mower pattern across the field, picking up balls at its
front sensor (ROBOT_RADIUS cells ahead in heading direction).  Keeps own-color
balls and spits opponent balls behind (inherited from RandomRobot).
"""

from __future__ import annotations

import numpy as np

from push_back.env.robots.base import BaseObservation
from push_back.env.robots.random_robot import Action, RandomRobot
from push_back.env.state import GRID_SIZE, ROBOT_RADIUS, Heading, Pose


class SweeperRobot(RandomRobot):
    """Sweeps the field row-by-row in a boustrophedon pattern."""

    _STUCK_THRESHOLD: int = 3  # skip waypoint after N FORWARD-but-didn't-move ticks

    def __init__(
        self,
        turn_ticks: int = 3,
        rng: np.random.Generator | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(turn_ticks=turn_ticks, rng=rng)
        self._debug: bool = debug
        self._tick: int = 0
        self._waypoints: list[tuple[int, int]] = []
        self._wp_index: int = 0
        self._initialized: bool = False
        self._last_pos: tuple[int, int] = (-1, -1)
        self._last_action: int = int(Action.STAY)
        self._stuck_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._waypoints = []
        self._wp_index = 0
        self._initialized = False
        self._last_pos = (-1, -1)
        self._last_action = int(Action.STAY)
        self._stuck_ticks = 0

    # ------------------------------------------------------------------
    # waypoint generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_waypoints() -> list[tuple[int, int]]:
        """Start-and-end waypoints per row for a boustrophedon sweep.

        Each row has two waypoints: the *start* (where the robot enters
        the row after transitioning) and the *end* (the far side).  This
        means skipping a single waypoint on stuck loses at most half a
        row of coverage rather than an entire row::

            min_c              max_c
              |                  |
              |  S→→→→→→→→→→→→E  y = min_c       wp 0-1
              |                ↓
              |  E←←←←←←←←←←←←S  y = min_c + 1   wp 2-3
              |  ↓
              |  S→→→→→→→→→→→→E  y = min_c + 2   wp 4-5
              |                ↓
              |  E←←←←←←←←←←←←S  y = min_c + 3   wp 6-7
              |  :
        """
        margin: int = ROBOT_RADIUS
        min_c: int = margin
        max_c: int = GRID_SIZE - 1 - margin

        waypoints: list[tuple[int, int]] = []
        going_right: bool = True
        for i, y in enumerate(range(min_c, max_c + 1)):
            if going_right:
                start_x, end_x = min_c, max_c
            else:
                start_x, end_x = max_c, min_c
            # First row: robot is already at the start edge, skip the
            # start waypoint so it doesn't pointlessly navigate to itself.
            if i > 0:
                waypoints.append((start_x, y))
            waypoints.append((end_x, y))
            going_right = not going_right
        return waypoints

    # ------------------------------------------------------------------
    # navigation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _desired_heading(pose: Pose, tx: int, ty: int) -> int | None:
        """Cardinal heading toward *(tx, ty)*, or ``None`` if already there.

        Prioritises x-axis alignment (the sweep direction) over y.
        """
        dx: int = tx - pose.x
        dy: int = ty - pose.y
        if dx > 0:
            return int(Heading.E)
        if dx < 0:
            return int(Heading.W)
        if dy > 0:
            return int(Heading.N)
        if dy < 0:
            return int(Heading.S)
        return None

    @staticmethod
    def _turn_toward(current_heading: int, desired_heading: int) -> int:
        """Return TURN_LEFT or TURN_RIGHT to minimise turning distance."""
        diff: int = (desired_heading - current_heading) % 8
        if diff <= 4:
            return int(Action.TURN_LEFT)
        return int(Action.TURN_RIGHT)

    # ------------------------------------------------------------------
    # tick
    # ------------------------------------------------------------------

    def tick(self, obs: BaseObservation, agent_id: int) -> int:
        pose = Pose(
            int(obs.agents[agent_id][0]),
            int(obs.agents[agent_id][1]),
            int(obs.agents[agent_id][2]),
        )

        if not self._initialized:
            self._waypoints = self._generate_waypoints()
            self._initialized = True

        # Stuck detection: count ticks where we sent FORWARD but didn't move.
        cur_pos: tuple[int, int] = (pose.x, pose.y)
        if cur_pos == self._last_pos and self._last_action == int(Action.FORWARD):
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
        self._last_pos = cur_pos

        if self._stuck_ticks >= self._STUCK_THRESHOLD and self._wp_index < len(
            self._waypoints
        ):
            self._wp_index += 1
            self._stuck_ticks = 0

        # Advance past reached waypoints.
        while self._wp_index < len(self._waypoints):
            tx, ty = self._waypoints[self._wp_index]
            if (pose.x, pose.y) != (tx, ty):
                break
            self._wp_index += 1

        # Navigate toward current waypoint.
        action: int = int(Action.STAY)
        if self._wp_index < len(self._waypoints):
            tx, ty = self._waypoints[self._wp_index]
            desired = self._desired_heading(pose, tx, ty)
            assert desired is not None
            if pose.heading == desired:
                action = int(Action.FORWARD)
            else:
                action = self._turn_toward(pose.heading, desired)

        if self._debug:
            self._tick += 1
            wp = (
                self._waypoints[self._wp_index]
                if self._wp_index < len(self._waypoints)
                else None
            )
            print(
                f"[Sweeper t={self._tick}] pos=({pose.x},{pose.y}) h={Heading(pose.heading).name}"
                f" wp#{self._wp_index}={wp}"
                f" stuck={self._stuck_ticks}"
                f" -> {Action(action).name}"
            )

        self._last_action = action
        return action
