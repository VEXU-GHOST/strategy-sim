from typing import Dict, List, Optional, Tuple, TypedDict


class RobotStrategicState(TypedDict):
    row: float
    col: float
    row_norm: float
    col_norm: float
    carrying: bool
    held_item_channels: List[int]
    busy: bool
    current_action: Optional[str]
    target: Optional[str]
    remaining_time: float


class ObjectStrategicState(TypedDict):
    object_id: str
    channel: int
    row: int
    col: int
    region: str
    count: int


class StrategicState(TypedDict):
    time: float
    time_remaining: float
    team_score: float
    agents: Dict[str, RobotStrategicState]
    object_states: List[ObjectStrategicState]
    goal_states: object
    toggle_states: object
    loader_states: object
