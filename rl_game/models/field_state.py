import numpy as np
import mcts_prototype

# 144 in. / 6 = 24, 6 in. x 6 in. tiles
FIELD_SIZE = 24

# Scoring:


class FieldState():

    def __init__(self):
        #self.ally_zones = np.zeros(FIELD_SIZE, FIELD_SIZE)
        self.curr_tasks = []
        #self.enemy_zones = np.zeros(FIELD_SIZE, FIELD_SIZE)
        self.likely_targets = np.zeros(FIELD_SIZE, FIELD_SIZE)
        self.cone_map = np.zeros(FIELD_SIZE, FIELD_SIZE)
        self.toggle_states = ['R', 'B', 'Y', 'G']
        self.high_val_obj = np.zeros(FIELD_SIZE, FIELD_SIZE)
        self.game_score = 0
        self.time_left = 120

    def tick_down(self):
        if self.time_left > 0:
            self.time_left -= 1

    def calc_score(self):
        goal_1 = self.toggle_states[6, 6]



#def update_state():