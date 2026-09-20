"""Render a simplified, animated version of the competition field in Manim.

Run from this folder so ``vid_inputs.json`` is found:
    manim -pqh manim_test.py VexSim
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from manim import *


CONFIG_PATH = Path(__file__).with_name("vid_inputs.json")
with CONFIG_PATH.open() as file:
    cfg = json.load(file)

RUNTIME = cfg["video"]["runtime"]
STEP = cfg["simulation"]["step_size"]
MIN_CMD = cfg["simulation"]["min_commands"]
MAX_CMD = cfg["simulation"]["max_commands"]
MIN_TURN = cfg["simulation"]["min_turn"]
MAX_TURN = cfg["simulation"]["max_turn"]
MIN_MOVE = cfg["simulation"]["min_move"]
MAX_MOVE = cfg["simulation"]["max_move"]

FIELD_HALF_WIDTH = cfg["field"]["half_width"]
FIELD_HALF_HEIGHT = cfg["field"]["half_height"]

DIR_MAP = {
    0: np.array([1.0, 0.0, 0.0]),
    1: np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
    2: np.array([0.0, 1.0, 0.0]),
    3: np.array([-1.0, 1.0, 0.0]) / np.sqrt(2),
    4: np.array([-1.0, 0.0, 0.0]),
    5: np.array([-1.0, -1.0, 0.0]) / np.sqrt(2),
    6: np.array([0.0, -1.0, 0.0]),
    7: np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
}


def field_point(point: list[float]) -> np.ndarray:
    """Convert a two-dimensional config point into Manim coordinates."""
    return np.array([point[0], point[1], 0.0])


def make_cup(center: np.ndarray, top_color: ManimColor) -> VGroup:
    """A cup is a circle split into a coloured top half and a white bottom."""
    radius = 0.28
    outline = Circle(radius=radius, color=BLACK, stroke_width=3)
    top = Arc(radius=radius, start_angle=0, angle=PI).set_fill(top_color, 1)
    top.set_stroke(width=0)
    bottom = Arc(radius=radius, start_angle=PI, angle=PI).set_fill(WHITE, 1)
    bottom.set_stroke(width=0)
    divider = Line(LEFT * radius, RIGHT * radius, color=BLACK, stroke_width=2)
    cup = VGroup(top, bottom, outline, divider)
    cup.move_to(center)
    return cup


def make_pin(center: np.ndarray, top_color: ManimColor, bottom_color: ManimColor) -> VGroup:
    """A compact hexagonal pin, visually close to the supplied field diagram."""
    top = Polygon(
        LEFT * 0.22 + UP * 0.02,
        UP * 0.22,
        RIGHT * 0.22 + UP * 0.02,
        RIGHT * 0.22 + DOWN * 0.10,
        LEFT * 0.22 + DOWN * 0.10,
        color=BLACK,
        stroke_width=3,
    ).set_fill(top_color, 1)
    bottom = Polygon(
        LEFT * 0.22 + DOWN * 0.10,
        RIGHT * 0.22 + DOWN * 0.10,
        RIGHT * 0.22 + DOWN * 0.30,
        DOWN * 0.43,
        LEFT * 0.22 + DOWN * 0.30,
        color=BLACK,
        stroke_width=3,
    ).set_fill(bottom_color, 1)
    pin = VGroup(top, bottom)
    pin.move_to(center)
    return pin


def make_robot(center: np.ndarray, body_color: ManimColor) -> VGroup:
    """A simple four-arm robot with a heading arrow."""
    core = Circle(radius=0.25, color=BLACK, stroke_width=3).set_fill(YELLOW, 1).shift(UP * 0.25)
    arms = VGroup(
        Triangle().scale(0.18).set_fill(body_color, 1).set_stroke(BLACK, 2).shift(UP * 0.44),
        Triangle().scale(0.18).rotate(PI).set_fill(body_color, 1).set_stroke(BLACK, 2).shift(DOWN * 0.44),
        Triangle().scale(0.18).rotate(-PI / 2).set_fill(body_color, 1).set_stroke(BLACK, 2).shift(LEFT * 0.44),
        Triangle().scale(0.18).rotate(PI / 2).set_fill(body_color, 1).set_stroke(BLACK, 2).shift(RIGHT * 0.44),
    )
    heading = Arrow(ORIGIN, RIGHT * 0.60, buff=0.18, color=WHITE, stroke_width=4).shift(UP * 0.25)
    robot = VGroup(arms, core, heading)
    robot.move_to(center)
    return robot


class VexSim(Scene):
    """Animate one robot moving around a visual model of the game field."""

    def build_field(self) -> tuple[VGroup, list[Circle]]:
        layout = cfg["layout"]
        field = VGroup()
        obstacles: list[Circle] = []

        board = Rectangle(
            width=2 * FIELD_HALF_WIDTH,
            height=2 * FIELD_HALF_HEIGHT,
            color=GREY_B,
            stroke_width=4,
        ).set_fill(GREY_D, opacity=0.88)
        field.add(board)

        for x in (-3, 0, 3):
            field.add(DashedLine([x, -FIELD_HALF_HEIGHT, 0], [x, FIELD_HALF_HEIGHT, 0], color=GREY_C, stroke_opacity=0.35))
        for y in (-3, 0, 3):
            field.add(DashedLine([-FIELD_HALF_WIDTH, y, 0], [FIELD_HALF_WIDTH, y, 0], color=GREY_C, stroke_opacity=0.35))

        for start, end in layout["diagonals"]:
            field.add(Line(field_point(start), field_point(end), color=WHITE, stroke_width=4))

        for start, end, color in layout["alliance_corners"]:
            field.add(Line(field_point(start), field_point(end), color=color, stroke_width=4))

        for start, end in layout["loaders"]:
            field.add(Line(field_point(start), field_point(end), color=YELLOW, stroke_width=8))

        for point in layout["gray_cups"]:
            cup = make_cup(field_point(point), GREY_D)
            field.add(cup)
            obstacles.append(Circle(radius=0.32).move_to(cup.get_center()))

        for point in layout["clear_cups"]:
            cup = make_cup(field_point(point), WHITE)
            field.add(cup)
            obstacles.append(Circle(radius=0.32).move_to(cup.get_center()))

        for point, colors in layout["pins"]:
            field.add(make_pin(field_point(point), colors[0], colors[1]))

        for point, color in layout["goals"]:
            field.add(Circle(radius=0.27, color=BLACK, stroke_width=3).set_fill(color, 1).move_to(field_point(point)))

        return field, obstacles

    def construct(self) -> None:
        self.camera.background_color = cfg["video"]["background_color"]
        field, cup_obstacles = self.build_field()
        self.play(FadeIn(field), run_time=1.0)

        robot_config = cfg["robot"]
        robot = make_robot(field_point(robot_config["start"]), robot_config["color"])
        self.play(FadeIn(robot), run_time=0.4)

        angle = 0
        position = robot.get_center().copy()
        command_count = random.randint(MIN_CMD, MAX_CMD)

        for _ in range(command_count):
            turn = random.randint(MIN_TURN, MAX_TURN)
            angle = (angle + 45 * turn) % 360
            self.play(robot.animate.rotate(45 * turn * DEGREES, about_point=robot.get_center()), run_time=RUNTIME)

            distance = random.randint(MIN_MOVE, MAX_MOVE) * STEP
            direction = DIR_MAP[(angle // 45) % 8]
            candidate = position + direction * distance

            inside_field = (
                -FIELD_HALF_WIDTH + 0.45 <= candidate[0] <= FIELD_HALF_WIDTH - 0.45
                and -FIELD_HALF_HEIGHT + 0.45 <= candidate[1] <= FIELD_HALF_HEIGHT - 0.45
            )
            hits_cup = any(np.linalg.norm(candidate[:2] - cup.get_center()[:2]) < 0.70 for cup in cup_obstacles)

            if inside_field and not hits_cup:
                self.play(robot.animate.move_to(candidate), run_time=RUNTIME, rate_func=linear)
                position = candidate
            else:
                self.play(Indicate(robot, color=RED), run_time=0.3)

        self.wait(1)
