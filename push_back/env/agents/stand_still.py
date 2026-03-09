"""StandStill agent — does nothing every tick."""

from __future__ import annotations

from push_back.env.state import Action, WorldState


class StandStill:
    """Returns STAY every tick. Useful as a placeholder."""

    def act(self, state: WorldState, agent_id: int) -> int:
        return Action.STAY
