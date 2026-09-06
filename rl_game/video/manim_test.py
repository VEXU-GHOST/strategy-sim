"""
IMPORT STATEMENTS:
Manim here is used for 3Blue1Brown-style animations and visualization.
The random library is for pseudorandom number generation and controls the robot translation and rotation.
Numpy is here for complicated mathematical operations.
The json library is used for taking in a JSON file with robot and field limtis as parameters.
"""
from manim import *
import random
import numpy as np
import json


"""
FILE INPUT:
Using json, this opens the file with the parameters.
"""
with open("vid_inputs.json") as f:
    cfg = json.load(f)


"""
CONSTANTS:
Runtime determines the .mp4 video file's length.
Steps are simple discrete time steps that advanced the simulation incrementally.
Boundary sets the field borders and stops the robot.
Mininum Commands is the lower boundary for the amount of instructions for the robot, with the resulting list length always even since each command appends two items.
Maximum Commands is the upper boundary for the amount of instructions for the robot, with the resulting list length always even since each command appends two items.
Mininum Turn is the lower boundary for the amount of rotation instructions for the robot.
Maximum Turn is the upper boundary for the amount of rotation instructions for the robot.
Mininum Move is the lower boundary for the amount of translation instructions for the robot.
Maximum Move is the upper boundary for the amount of translation instructions for the robot.
"""
RUNTIME  = cfg["video"]["runtime"]
STEP     = cfg["simulation"]["step_size"]
BOUNDARY = cfg["simulation"]["boundary"]
MIN_CMD  = cfg["simulation"]["min_commands"]
MAX_CMD  = cfg["simulation"]["max_commands"]
MIN_TURN = cfg["simulation"]["min_turn"]
MAX_TURN = cfg["simulation"]["max_turn"]
MIN_MOVE = cfg["simulation"]["min_move"]
MAX_MOVE = cfg["simulation"]["max_move"]


"""
DIRECTION MAP:
Simply the cardinal directions with their diagonals.
East
Northeast
North
Northwest
West
Southwest
South
Southeast
"""
DIR_MAP = {
    0: np.array([ 1,  0, 0]),
    1: np.array([ 1,  1, 0]),
    2: np.array([ 0,  1, 0]),
    3: np.array([-1,  1, 0]),
    4: np.array([-1,  0, 0]),
    5: np.array([-1, -1, 0]),
    6: np.array([ 0, -1, 0]),
    7: np.array([ 1, -1, 0]),
}


"""
MANIM SCENE CLASS:
This makes the overall Manim scene using one function and a lot of logic.
"""
class VexSim(Scene):

    # Function here is for setting all variables and objects, then running their respective logic.
    def construct(self):

        # This reads from the JSON file for all things under "field".
        # The field will take the form of a numberplane.
            # This sets the field width positions the robot can take.
            # This sets the field height positions the robot can take.
            # This sets the rendered field's horizontal squares.
            # This sets the rendered field's vertical squares.
        # The animation for drawing the field is played.
        # A small wait happens.
        fc = cfg["field"]
        field = NumberPlane(
            x_range = fc["x_range"],
            y_range = fc["y_range"],
            x_length = fc["x_length"],
            y_length = fc["y_length"]
        )
        self.play(Create(field))
        self.wait()

        # This reads from the JSON file for all things under "robot".
        # The robot body takes the form of a circle of a certain radius, color, and opacity.
        # The bot's direction it is facing, represented as a line, is intialized as the rightward (east) direction.
        # The bot is positioned at a pair of x and y coordinates.
        # A Mobject for the robot is actualized.
        # Bot Mobject is added.
        rc = cfg["robot"]
        body = Circle(radius = rc["radius"]).set_fill(rc["color"], opacity = rc["opacity"])
        heading = Line(ORIGIN, RIGHT * rc["heading_length"]).set_color(rc["heading_color"])
        start = np.array([rc["start_x"], rc["start_y"], 0.0])
        robot = VGroup(body, heading).move_to(start)
        self.add(robot)

        # An empty list for instructions is initialized.
        # A certain amount of robot commands is generated.
        # The for loop is initialized, running for a number of iterations equal to the number of commands.
            # The bot will turn a certain angle.
            # That rotation is appended to the list.
            # The bot will move a certain distance.
            # That translation is appended to the list.
        # A small print statement for debugging.
        instrs = []
        cmd = random.randint(MIN_CMD, MAX_CMD)
        for i in range(cmd):
            turn = random.randint(MIN_TURN, MAX_TURN)
            instrs.append(turn)
            move = random.randint(MIN_MOVE, MAX_MOVE)
            instrs.append(move)
        print(instrs)

        # The bot's angle is intialized as zero.
        # The position from before is copied over for the robot's starting place.
        angle = 0
        pos = start.copy()

        # Another for loop here for running through each instruction command.
            # An even index in the list is a rotation.
                # Turns are snapped to any multiple of 45 degrees.
                # The next angle is set as a multiple of 45.
            # An odd index in the list is a translation.
                # Direction is looked up from the direction map based on the current heading angle.
                # The new position is based off the old plus the product of a direction vector, a spatial step size, and a move scalar.
                # All positions are clamped by the borders.
                # The x position is clamped upon collision.
                # The y position is clamped upon collision.
        for i, instr in enumerate(instrs):
            if i % 2 == 0:
                self.play(robot.animate.rotate(45 * instr * DEGREES), rate_func=linear, run_time=RUNTIME)
                angle += 45 * instr
            else:
                direction = DIR_MAP[(angle // 45) % 8]
                new_pos = pos + direction * STEP * instr
                clamped = new_pos.copy()
                clamped[0] = np.clip(round(clamped[0]), -BOUNDARY, BOUNDARY)
                clamped[1] = np.clip(round(clamped[1]), -BOUNDARY, BOUNDARY)

                # Only move if the new position is within bounds.
                    # The animation for moving the bot plays.
                    # The robot is transported.
                if np.array_equal(new_pos, clamped):
                    self.play(robot.animate.move_to(new_pos), rate_func=linear, run_time=RUNTIME)
                    pos = new_pos