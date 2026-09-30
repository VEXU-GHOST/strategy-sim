import numpy as np
import mcts_prototype
import sys

# 144 in. / 24 = 6, 24 in. x 24 in. tiles
FIELD_SIZE = 6

# Scoring:


class FieldState():

    # Everything commented out here is to be implemented into a later version, only doing map, score, and time for now.
    def __init__(self):
        # self.ally_zones = np.zeros(FIELD_SIZE, FIELD_SIZE)
        self.curr_tasks = []
        # self.enemy_zones = np.zeros(FIELD_SIZE, FIELD_SIZE)
        # self.likely_targets = np.full(shape=(FIELD_SIZE, FIELD_SIZE), fill_value="E")
        self.map = np.full(shape=(FIELD_SIZE, FIELD_SIZE), fill_value="E") 
        self.map[4, 1] = "C" # cone
        self.map[1, 4] = "G" # goal
        self.map[0, 0] = "B" # bot
        # self.toggle_states = ['R', 'B', 'Y', 'G']
        # self.high_val_obj = np.full(shape=(FIELD_SIZE, FIELD_SIZE), fill_value=0) 
        self.game_score = 0
        self.time_left = 120

    def tick_down(self):
        if self.time_left > 0:
            self.time_left -= 1

    def calc_score(self):
        goal_1 = ...

    def update_state(self):
        # stuff that can be done in under a second
        self.tick_down()


# test main
def main():
    field = FieldState()
    np.set_printoptions(linewidth=200)
    print(field.map)
    while field.time_left > 0:
        field.update_state()
    print(field.game_score)

if __name__ == "__main__": 
    main()