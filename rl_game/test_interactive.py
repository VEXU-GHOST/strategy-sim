"""
Interactive testing script — play one episode with user input.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from env.game_env import GameEnv
from agents.dqn_agent import DQNAgent
import numpy as np

def print_game_grid(state):
    """Print a simple ASCII view of the current board state."""
    size = config.BOARD_SIZE
    grid = [["." for _ in range(size)] for _ in range(size)]

    channel_symbols = {
        1: "A",
        2: "B",
        3: "*",
        4: "o",
    }

    val = np.zeros((35, 35))
    for i in range(35):
        for j in range(35):
             for k in range(5):
                 val[i][j] +=state["state"][k][i][j]
            #val[i][j]+=state["state"][4][i][j]

    for row in range(35):
        
        
        for col in range(35):
            symbol = " "
            # if state["pos"]["A0"][0]==row and state["pos"]["A0"][1]==col:
            #     symbol = "A"
            # elif state["pos"]["A1"][0]==row and state["pos"]["A1"][1]==col:
            #     symbol = "A"
            # elif state["pos"]["B0"][0]==row and state["pos"]["B0"][1]==col:
            #     symbol = "B"
            # elif state["pos"]["B1"][0]==row and state["pos"]["B1"][1]==col:
            #     symbol = "B"
            if val[row][col]!=0:
            #if True:
                symbol = str(int(val[row][col]))
            if row == 200 or col == 200:
                continue
            if 0 <= row < size and 0 <= col < size:
                grid[row][col] = symbol

    agent_symbols = {"A0": "a", "A1": "A", "B0": "b", "B1": "B"}
    for agent_id, (row, col) in state["pos"].items():
        if 0 <= row < size and 0 <= col < size:
            grid[row][col] = agent_symbols.get(agent_id, agent_id[0])

    print("\nGame grid:")
    for row in grid:
        print(" ".join(row))


def test_interactive():
    """Run a single episode with user input for Team B."""
    env = GameEnv()
    state = env.reset()
    drawState = {
        "state": env._board.copy(),
        "pos": {aid: (a.row, a.col) for aid, a in env._agents.items()}
    }
    print_game_grid(drawState)

    # Create agents
    team_a = DQNAgent(team=config.TEAM_A, agent_id="Team_A")
    team_b = DQNAgent(team=config.TEAM_B, agent_id="Team_B")

    # Load trained models if available
    if os.path.exists("checkpoints/team_a_final.pth"):
        team_a.load("checkpoints/team_a_final.pth")
        print("✓ Loaded Team A checkpoint")

    if os.path.exists("checkpoints/team_b_final.pth"):
        team_b.load("checkpoints/team_b_final.pth")
        print("✓ Loaded Team B checkpoint")
    
    done = False
    step = 0
    
    print("\n" + "="*50)
    print("INTERACTIVE TEST MODE")
    print("="*50)
    print("Team A: AI (trained agent)")
    print("Team B: USER INPUT")
    print("="*50 + "\n")
    
    while not done and step < config.STEPS_PER_EPISODE:
        print(f"\n--- STEP {step} ---")

        # Team A uses AI
        # action_a = team_a.act(state, user_input=False, epsilon=0.0)
        # print(f"Team A action: ({action_a})")

        # Team B uses user input
        print("\nTeam B - Enter your move:")
        print(str(env._agents["B1"].row)+" "+str(env._agents["B1"].row))
        print("  action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=UP-LEFT, 5=UP-RIGHT, 6=DOWN-LEFT, 7=DOWN-RIGHT, 8=STAY")

        try:
            action_b = int(input("  action: "))

            actions = {
                "A0": 0, "A1": 0,  # Both Team A members use same action
                "B0": action_b, "B1": action_b,  # Both Team B members use same action
            }
            
            state, rewards, done, info = env.step(actions)
            drawState = {
                "state": env._board.copy(),
                "pos": {aid: (a.row, a.col) for aid, a in env._agents.items()}
            }
            print_game_grid(drawState)
            print(f"\nRewards: {rewards}")
            print(f"Done: {done}")
            step += 1
            
        except ValueError:
            print("Invalid input! Please enter integers.")
            continue
    
    print("\n" + "="*50)
    print("Episode finished!")
    print("="*50)

if __name__ == "__main__":
    test_interactive()
