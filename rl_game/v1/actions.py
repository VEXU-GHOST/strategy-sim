from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    COLLECT = "collect"
    SCORE = "score"
    TOGGLE = "toggle"
    WAIT = "wait"


@dataclass(frozen=True)
class MacroAction:
    action_type: ActionType
    target: Optional[str] = None

    def __str__(self) -> str:
        if self.target is None:
            return self.action_type.value
        return f"{self.action_type.value}({self.target})"
