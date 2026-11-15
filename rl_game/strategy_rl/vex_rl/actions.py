"""Stable discrete action-ID space, used by PPO, heuristics, and future adapters."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

KINDS = ("alliance_pin", "yellow_pin", "cup")
BLOCK_LOCATIONS = ("midfield", "north", "south", "center_goal")


@dataclass(frozen=True, slots=True)
class Action:
    verb: str
    target: str = ""
    object_type: str = ""

    def __str__(self) -> str:
        return f"{self.verb}({', '.join(x for x in (self.target, self.object_type) if x)})"


class ActionCodec:
    def __init__(self, config: dict):
        self.actions: list[Action] = []
        for source in config["sources"]:
            for kind in KINDS:
                self.actions.append(Action("COLLECT", source["id"], kind))
        for goal in config["goals"]:
            for kind in KINDS:
                self.actions.append(Action("SCORE", goal["id"], kind))
        for toggle in config["toggles"]:
            self.actions.append(Action("SET_TOGGLE", toggle["id"]))
        self.actions.extend([Action("GO_MIDFIELD"), Action("WAIT")])
        for location in BLOCK_LOCATIONS:
            self.actions.append(Action("BLOCK", location))
        self._index = {a: i for i, a in enumerate(self.actions)}
        assert len(self._index) == len(self.actions)
        # Canonical action IDs are red-side IDs. Frozen PPO on the blue side
        # remaps every target to its 180-degree symmetric counterpart.
        symmetry = config["symmetry"]
        self._blue_permutation = []
        for action in self.actions:
            group = {"COLLECT": "sources", "SCORE": "goals", "SET_TOGGLE": "toggles"}.get(action.verb)
            corresponding = (Action(action.verb, symmetry[group][action.target], action.object_type)
                             if group is not None else action)
            self._blue_permutation.append(self._index[corresponding])
        assert sorted(self._blue_permutation) == list(range(len(self.actions)))

    def __len__(self):
        return len(self.actions)

    def decode(self, index: int) -> Action:
        if not 0 <= int(index) < len(self.actions):
            raise ValueError(f"action ID out of range: {index}")
        return self.actions[int(index)]

    def encode(self, action: Action) -> int:
        return self._index[action]

    def describe(self, index: int) -> str:
        return str(self.decode(index))

    def available(self, mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask)

    def canonical_mask(self, physical_mask: np.ndarray, team: int) -> np.ndarray:
        return (physical_mask if team == 0 else
                physical_mask[np.asarray(self._blue_permutation)]).copy()

    def physical_index(self, canonical_action_id: int, team: int) -> int:
        self.decode(canonical_action_id)  # range validation
        return (canonical_action_id if team == 0 else
                self._blue_permutation[canonical_action_id])
