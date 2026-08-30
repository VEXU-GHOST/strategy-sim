from manim import *
import secrets
import numpy as np

RUNTIME = 0.5

DIR_MAP = {
    0: np.array([ 1,  0, 0]),  # Right
    1: np.array([ 1,  1, 0]),  # NE
    2: np.array([ 0,  1, 0]),  # Up
    3: np.array([-1,  1, 0]),  # NW
    4: np.array([-1,  0, 0]),  # Left
    5: np.array([-1, -1, 0]),  # SW
    6: np.array([ 0, -1, 0]),  # Down
    7: np.array([ 1, -1, 0]),  # SE
}

class VexSim(Scene):
    def construct(self):

        # This draws the field.
        field = NumberPlane(
            x_range=(-5, 5, 1),
            y_range=(-5, 5, 1),
            x_length = 5,
            y_length = 5
        )
        self.play(Create(field))
        self.wait()

        # This draws the robot.
        body = Circle(radius = 0.5).set_fill(BLUE, opacity = 1)
        heading = Line(ORIGIN, RIGHT * 0.5, buff = 0).set_color(YELLOW)
        robot = VGroup(body, heading).move_to(LEFT * 1)  # matches pos below
        self.add(robot)

        # This generates a list of instructions for the robot to move to.
        instrs = []
        cmd = secrets.SystemRandom().randint(1, 10)
        for i in range(cmd):
            turn = secrets.SystemRandom().randint(-4, 4)
            instrs.append(turn)
            move = secrets.SystemRandom().randint(-4, 4)
            instrs.append(move)
        print(instrs)

        # This moves the robot.
        angle = 0
        pos = np.array([-1.0, 0.0, 0.0])  # matches move_to(LEFT * 1)
        STEP = 1
        BOUNDARY = 2.0

        for i, instr in enumerate(instrs):
            if i % 2 == 0:  # even index = turn
                self.play(robot.animate.rotate(45 * instr * DEGREES), rate_func=linear, run_time=RUNTIME)
                angle += 45 * instr
            else:           # odd index = move
                direction = DIR_MAP[(angle // 45) % 8]
                new_pos = pos + direction * STEP * instr
                clamped = new_pos.copy()
                clamped[0] = np.clip(round(clamped[0]), -BOUNDARY, BOUNDARY)
                clamped[1] = np.clip(round(clamped[1]), -BOUNDARY, BOUNDARY)

                if np.array_equal(new_pos, clamped):  # no clamping needed
                    self.play(robot.animate.move_to(new_pos), rate_func=linear, run_time=RUNTIME)
                    pos = new_pos
                # else: out of bounds, do nothing