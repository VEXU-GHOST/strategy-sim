from __future__ import annotations

from typing import List

from actions import ActionType, MacroAction


def get_available_actions(state, agent_id: str, bridge) -> List[MacroAction]:
    """
    Return only high-level actions that V1 can safely execute.

    SCORE and TOGGLE remain in the global action interface, but are not
    exposed until their full GameEnv execution mappings are implemented.
    """
    agent_state = state["agents"][agent_id]

    if agent_state["busy"]:
        return []

    actions: List[MacroAction] = []

    if not agent_state["carrying"]:
        for region in bridge.collect_targets():
            actions.append(MacroAction(ActionType.COLLECT, region))

    # Do not expose SCORE yet: current GameEnv scoring internals are still WIP.
    # Once GameEnvBridge.apply_macro_action supports SCORE, add goal actions here.

    # Do not expose TOGGLE yet: current GameEnv does not expose reliable
    # toggle coordinates for duration/execution mapping.

    actions.append(MacroAction(ActionType.WAIT, None))
    return actions
