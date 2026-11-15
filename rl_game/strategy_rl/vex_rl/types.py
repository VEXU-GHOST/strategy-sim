"""Small mutable state objects for the semi-Markov macro-action simulator."""
from __future__ import annotations
from dataclasses import dataclass, field
from .actions import Action


@dataclass(slots=True)
class Pending:
    action: Action
    finish_at: float
    destination: tuple[float, float]


@dataclass(slots=True)
class Robot:
    name: str
    team: int
    position: tuple[float, float]
    pin: str | None = None
    cup: bool = False
    pending: Pending | None = None


@dataclass(slots=True)
class StackItem:
    kind: str  # red_pin, blue_pin, yellow_pin, cup
    team: int


@dataclass(slots=True)
class Goal:
    goal_id: str
    position: tuple[float, float]
    quadrant: int
    owner: str
    max_items: int
    stack: list[StackItem] = field(default_factory=list)

    def can_add(self, kind: str) -> bool:
        if len(self.stack) >= self.max_items:
            return False
        if kind == "cup":
            return bool(self.stack and self.stack[-1].kind != "cup")
        return not self.stack or self.stack[-1].kind == "cup"


@dataclass(slots=True)
class Source:
    source_id: str
    position: tuple[float, float]
    available: dict[str, int]
