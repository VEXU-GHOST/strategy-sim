"""Tests for the collision system."""

from push_back.env.collision import resolve_moves
from push_back.env.state import Pose, ROBOT_RADIUS


def test_two_robots_move_to_same_cell() -> None:
    """Two robots heading toward the same cell — both should be rejected."""
    #  Robot 0 at (10, 10), Robot 1 at (10, 14).
    #  Both propose to move to (10, 12).
    current = [
        Pose(10, 10, 0),
        Pose(10, 14, 0),
    ]
    proposed = [
        Pose(10, 12, 0),
        Pose(10, 12, 0),
    ]
    result = resolve_moves(current, proposed, blocked_cells=set(), robot_radius_cells=ROBOT_RADIUS)

    # Both should revert to their original positions
    assert result[0].x == 10 and result[0].y == 10
    assert result[1].x == 10 and result[1].y == 14


def test_two_robots_within_radius_both_revert() -> None:
    """Two robots whose proposed positions overlap within ROBOT_RADIUS — both revert."""
    # Robot radius is 2, so diameter is 4. Distance of 3 < 4 → overlap.
    current = [
        Pose(10, 10, 0),
        Pose(10, 16, 0),
    ]
    proposed = [
        Pose(10, 11, 0),
        Pose(10, 13, 0),
    ]
    result = resolve_moves(current, proposed, blocked_cells=set(), robot_radius_cells=ROBOT_RADIUS)

    # Distance between proposed = 2, which is < 2*ROBOT_RADIUS=4 → overlap
    assert result[0].x == 10 and result[0].y == 10
    assert result[1].x == 10 and result[1].y == 16


def test_robots_far_apart_both_move() -> None:
    """Two robots far enough apart should both move successfully."""
    current = [
        Pose(5, 5, 0),
        Pose(20, 20, 0),
    ]
    proposed = [
        Pose(6, 5, 0),
        Pose(21, 20, 0),
    ]
    result = resolve_moves(current, proposed, blocked_cells=set(), robot_radius_cells=ROBOT_RADIUS)

    assert result[0].x == 6 and result[0].y == 5
    assert result[1].x == 21 and result[1].y == 20


def test_move_into_blocked_cell_reverts() -> None:
    """Robot trying to move into a blocked cell stays put, keeps heading."""
    blocked = {(11, 10)}
    current = [Pose(10, 10, 0)]
    proposed = [Pose(11, 10, 0)]
    result = resolve_moves(current, proposed, blocked_cells=blocked, robot_radius_cells=ROBOT_RADIUS)

    assert result[0].x == 10 and result[0].y == 10
    assert result[0].heading == 0  # heading preserved from proposed


def test_heading_preserved_on_collision_revert() -> None:
    """When a move is reverted, the proposed heading is still applied."""
    current = [
        Pose(10, 10, 0),
        Pose(10, 14, 0),
    ]
    proposed = [
        Pose(10, 12, 2),  # turned heading to N
        Pose(10, 12, 6),  # turned heading to S
    ]
    result = resolve_moves(current, proposed, blocked_cells=set(), robot_radius_cells=ROBOT_RADIUS)

    # Positions revert, but headings come from proposed
    assert result[0].x == 10 and result[0].y == 10
    assert result[0].heading == 2
    assert result[1].x == 10 and result[1].y == 14
    assert result[1].heading == 6
