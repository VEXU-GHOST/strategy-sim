import field_state
import random
import numpy as np

class Agent():

    def check_around_then_move(field_state.map):

        curr_pos = np.argwhere(field_state.map == "B")[0]
        rewards = [0, 0, 0, 0]

        # up, right, down, left
        # 1 for column, 0 for row
        dir = ((1, 1), (0, 1), (1, -1), (0, -1))
        for i in range(4):
            if field_state[curr_pos[dir[i][0]] + dir[i][1]] == "C":
                rewards[i] += 2 # upon seeing cone
            elif field_state[curr_pos[dir[i][0]] + dir[i][1]] == "E":
                rewards[i] += 1 # upon seeing empty space
            else:
                rewards[i] += 0 # upon seeing something else like a border

        max_reward = max(rewards)
        max_rewards = [i for i, val in enumerate(rewards) if val == max_reward]
        pref_dir_idx = random.choice(max_rewards)
        new_pos = curr_pos
        if dir[pref_dir_idx][0] == 1:
            new_pos = [curr_pos[0]][curr_pos[1] + dir[pref_dir_idx][1]]
        else:
            new_pos = [curr_pos[0] + dir[pref_dir_idx][1]][curr_pos[1]]
        field_state.map[new_pos[0]][new_pos[1]] = "B"
        field_state.map[curr_pos[0]][curr_pos[1]] = "E"
        curr_pos = new_pos