"""
env/game_env.py — 64×64 grid game environment.

Replace the stub logic in _apply_actions(), _compute_rewards(), and
_check_done() with your actual game rules.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BOARD_SIZE,
    NUM_CHANNELS,
    Action,
    TEAM_A,
    TEAM_B,
    SPAWN_POSITIONS,
    STEPS_PER_EPISODE,
)


# Shorthand type aliases
State    = np.ndarray          # shape (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
AgentID  = str                 # "A0", "A1", "B0", "B1"
Actions  = Dict[AgentID, int]  # {agent_id: action_int}
Rewards  = Dict[AgentID, float]


class AgentState:
    """Tracks a single agent's mutable position and alive status."""
    
    def __init__(self, agent_id: AgentID, team: int, row: int, col: int):
        self.agent_id = agent_id
        self.team = team
        self.row = row
        self.col = col
        self.held_items = np.zeros(5)
        self.stuck = False
    
    def move(self, action: int, drop: int, pick: int, board_size: int) -> tuple[int, int]:
        """Apply movement action, clamped to board boundaries."""
        if self.stuck:
            self.stuck = False
            return -1, -1
        if action == Action.UP:
            self.row = max(0, self.row - 1)
        elif action == Action.DOWN:
            self.row = min(board_size - 1, self.row + 1)
        elif action == Action.LEFT:
            self.col = max(0, self.col - 1)
        elif action == Action.RIGHT:
            self.col = min(board_size - 1, self.col + 1)
        elif action == Action.U_L:
            self.row = max(0, self.row - 1)
            self.col = max(0, self.col - 1)
        elif action == Action.U_R:
            self.row = max(0, self.row - 1)
            self.col = min(board_size - 1, self.col + 1)
        elif action == Action.D_L:
            self.row = min(board_size - 1, self.row + 1)
            self.col = max(0, self.col - 1)
        elif action == Action.D_R:
            self.row = min(board_size - 1, self.row + 1)
            self.col = min(board_size - 1, self.col + 1)
        
        if drop >= 0:
            if self.held_items[drop]>0:
                self.held_items[drop] = 0
            else:
                drop = -1;
        if action == Action.STAY:
            if 0 <= pick < len(self.held_items):
                self.held_items[pick] = 1
        return drop, pick
            
    def scoreGoal(self, score: int) -> int: #0:y 1:r 2:b 3:gray 4:clear
        if (score>2 and self.held_items[4]==1):
            if score==3:
                return 4
            else:
                return 3
        pin = -1
        for i in range(0,3):
            if(self.held_items[i]==1):
                pin = i
        # print(str(self.held_items))
        grid = [0,0,0,0,0]
        if(pin == 0):
            grid[0] = 1
            grid[2] = 1
        if(pin == 1):
            grid[0] = 1
            grid[1] = 1
        if(pin == 2):
            grid[1] = 1
            grid[2] = 1
        if(pin == 3):
            grid[0] = 1
            grid[0] = 1
        if(pin == 4):
            grid[3] = 1
            grid[4] = 1
        if grid[score]!=1:
            return -1
        for i in range(0,4):
            if grid[i] == 1:
                return i
        
        
            
            
        # STAY do not move the agent


class GameEnv:
    """
    64×64 multi-agent grid environment.

    Board tensor channels:
        Channel 0: empty cells (always 1 where no agent/object exists)
        Channel 1: Team-A agent positions (value = 1)
        Channel 2: Team-B agent positions (value = 1)
        Channel 3: Reserved for game objects / terrain (customize freely)

    Usage:
        env = GameEnv()
        state = env.reset()
        next_state, rewards, done, info = env.step(actions)
    """

    AGENT_IDS: list[AgentID] = ["A0", "A1", "B0", "B1"]

    def __init__(self):
        self._board_size = BOARD_SIZE
        self.AGENT_RADIUS = 2
        self._agents: Dict[AgentID, AgentState] = {}
        self._step_count: int = 0
        self._board: np.ndarray = np.zeros(
            (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        self.redMatchload = np.zeros(5)
        self.blueMatchload = np.zeros(5)
        self.thingsHeld = np.zeros((4, 5))
        self.zoneColors = np.zeros(4); #zones, colors
        self.weightUpdates = np.zeros(4);



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset the environment and return the initial board state."""
        self._step_count = 0
        self._board = np.zeros(
            (NUM_CHANNELS, self._board_size, self._board_size), dtype=np.int32
        )
        self.redMatchload[1] = 10
        self.redMatchload[3] = 1
        self.redMatchload[4] = 10
        self.blueMatchload[2] = 10
        self.blueMatchload[3] = 1
        self.blueMatchload[4] = 10
        # 0:b/y   1:r/y    2:r/b   3:y/y   4: cup
        self._board[3][5][5] = 1
        self._board[3][29][29] = 1
        self._board[3][11][11] = 1
        self._board[3][23][23] = 1
        self._board[3][17][17] = 1# mid diagonal
        self._board[3][11][0] = 1
        self._board[3][0][11] = 1
        self._board[3][23][34] = 1
        self._board[3][34][23] = 1
        self._board[3][23][0] = 1
        self._board[3][0][23] = 1
        self._board[3][11][34] = 1
        self._board[3][34][11] = 1# half corner
        self._board[3][23][5] = 1   
        self._board[3][29][11] = 1
        self._board[3][5][23] = 1
        self._board[3][11][29] = 1# non main diagonal
        self._board[2][17][11] = 1
        self._board[2][17][23] = 1
        self._board[2][11][17] = 1
        self._board[2][23][17] = 1# r/b
        self._board[1][29][5] = 2
        self._board[1][5][29] = 2
        self._board[1][23][11] = 2
        self._board[1][11][23] = 2# r/y
        self._board[0][29][5] = 2
        self._board[0][5][29] = 2
        self._board[0][23][11] = 2
        self._board[0][11][23] = 2# b/y
        self._board[4][5][5] = 1
        self._board[4][29][29] = 1
        self._board[4][23][23] = 1
        self._board[4][11][11] = 1
        self._board[4][11][0] = 3
        self._board[4][0][11] = 3
        self._board[4][23][0] = 3
        self._board[4][0][23] = 3
        self._board[4][11][34] = 3
        self._board[4][34][11] = 3
        self._board[4][23][34] = 3
        self._board[4][34][23] = 3
        self._board[4][5][29] = 1
        self._board[4][29][5] = 1
        self._board[4][11][23] = 1
        self._board[4][23][11] = 1

        self._board[4][17][11] = 1
        self._board[4][17][23] = 1
        self._board[4][11][17] = 1
        self._board[4][23][17] = 1#cups
        self.fieldObject = np.array([
            [5, 11, 0],
            [11, 5, 0],
            [23, 29, 0],
            [29, 23, 0],
            [17, 17, 0],
            [23, 5, 0],
            [29, 11, 0],
            [5, 23, 0],
            [11, 29, 0],
        ], dtype=int)
        self.goalStates = np.array([
            [0,0,0,3],
            [0,0,0,3],
            [0,0,0,3],
            [0,0,0,3],
            [1,0,0,0],
            [1,0,0,0],
            [1,0,0,0],
            [1,0,0,0],
            [1,0,0,0],
        ])
        self.points = {aid: 0.0 for aid in self.AGENT_IDS}
        # Spawn agents at configured positions
        self._agents = {}
        for agent_id, (row, col) in SPAWN_POSITIONS.items():
            team = TEAM_A if agent_id.startswith("A") else TEAM_B
            self._agents[agent_id] = AgentState(agent_id, team, row, col)

        self._rebuild_board()
        array0 = [(0, 0)]*20
        array1 = [(0, 0)]*20
        array2 = [(0, 0)]*4
        array3 = [(0, 0)]*19
        array4 = [(0, 0)]*56
        a0 = a1 = a2 = a3 = a4 = 0
        for row in range(len(self._board[1])):
            for col in range(len(self._board[1][row])):
                for i in range(len(self._board)):
                    val = self._board[i][row][col]
                    for j in range(val):
                        if(i==0):
                            array0[a0] = [row, col]
                            a0+=1
                        if(i==1):
                            array1[a1] = [row, col]
                            a1+=1
                        if(i==2):
                            array2[a2] = [row, col]
                            a2+=1
                        if(i==3):
                            array3[a3] = [row, col]
                            a3+=1
                        if(i==4):
                            array4[a4] = [row, col]
                            a4+=1
        while a0<20:
            array0[a0] = [200, 200]
            a0+=1
        while a1<20:
            array1[a1] = [200, 200]
            a1+=1
        while a2<4:
            array2[a2] = [200, 200]
            a2+=1
        while a3<19:
            array3[a3] = [200, 200]
            a3+=1
        while a4<56:
            array4[a4] = [200, 200]
            a4+=1

        finArray = [array0, array1, array2, array3, array4]
        state = {
            "board": finArray,
            "goalStates": self.goalStates.copy(),
            "loaderStates": [self.redMatchload.copy(), self.blueMatchload.copy()],
            "zoneColors": self.zoneColors.copy(),
            "agent_positions": {aid: (a.row, a.col) for aid, a in self._agents.items()}
        }

        return state

    def step(self, actions: Actions) -> Tuple[dict, Rewards, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Args:
            actions: Dict mapping agent_id → action int for all 4 agents.

        Returns:
            next_state : board tensor after this step
            rewards    : dict of per-agent float rewards
            done       : True if episode is finished
            info       : extra diagnostic information
        """
        red, blue = self.calcGoalScore()

        # print(str(self._agents["B1"].row)+" "+str(self._agents["B1"].col))
        self._step_count += 1


        # 1. Apply movement / actions
        self._apply_actions(actions)

        # 2. Rebuild board tensor from new agent positions
        self._rebuild_board()

        done = self._check_done()
        # 3. Compute rewards
        rewards = self._compute_rewards(actions, red, blue, done)

        # 4. Check termination
       

        array0 = [(0, 0)]*20
        array1 = [(0, 0)]*20
        array2 = [(0, 0)]*4
        array3 = [(0, 0)]*19
        array4 = [(0, 0)]*56
        a0 = a1 = a2 = a3 = a4 = 0
        for row in range(len(self._board[1])):
            for col in range(len(self._board[1][row])):
                for i in range(len(self._board)):
                    val = self._board[i][row][col]
                    for j in range(val):
                        if(i==0):
                            array0[a0] = [row, col]
                            a0+=1
                        if(i==1):
                            array1[a1] = [row, col]
                            a1+=1
                        if(i==2):
                            array2[a2] = [row, col]
                            a2+=1
                        if(i==3):
                            array3[a3] = [row, col]
                            a3+=1
                        if(i==4):
                            array4[a4] = [row, col]
                            a4+=1
        while a0<20:
            array0[a0] = [200, 200]
            a0+=1
        while a1<20:
            array1[a1] = [200, 200]
            a1+=1
        while a2<4:
            array2[a2] = [200, 200]
            a2+=1
        while a3<19:
            array3[a3] = [200, 200]
            a3+=1
        while a4<56:
            array4[a4] = [200, 200]
            a4+=1

        finArray = [array0, array1, array2, array3, array4]

        info = {
            "step": self._step_count,
            "agent_positions": {aid: (a.row, a.col) for aid, a in self._agents.items()},
        }
        state = {
            "board": finArray.copy(),
            "goalStates": self.goalStates.copy(),
            "loaderStates": [self.redMatchload.copy(), self.blueMatchload.copy()],
            "zoneColors": self.zoneColors.copy(),
            "agent_positions": {aid: (a.row, a.col) for aid, a in self._agents.items()},
        }
        # print(str(self._agents["B1"].row)+" "+str(self._agents["B1"].col))
        return state, rewards, done, info

    def render(self, mode: str = "ascii") -> None:
        """
        Simple ASCII render of the board (agents only).
        Replace or extend for richer visualisation.
        """
        grid = [["." for _ in range(self._board_size)]
                for _ in range(self._board_size)]

        symbols = {"A0": "a", "A1": "A", "B0": "b", "B1": "B"}
        for agent_id, agent in self._agents.items():
            grid[agent.row][agent.col] = symbols[agent_id]

        # Print a 16×16 sub-sample for readability
        sample = 16
        step = self._board_size // sample
        print(f"\n--- Step {self._step_count} ---")
        for r in range(0, self._board_size, step):
            print("".join(grid[r][c] for c in range(0, self._board_size, step)))

    @property
    def agent_ids(self) -> list[AgentID]:
        return self.AGENT_IDS

    @property
    def board_shape(self) -> Tuple[int, ...]:
        return self._board.shape

    # ------------------------------------------------------------------
    # Internal helpers — replace stub logic below with real game rules
    # ------------------------------------------------------------------


    def _safe_board_value(self, channel: int, row: int, col: int) -> int:
        if 0 <= channel < self._board.shape[0] and 0 <= row < self._board.shape[1] and 0 <= col < self._board.shape[2]:
            return int(self._board[channel, row, col])
        return 0

    def _safe_board_add(self, channel: int, row: int, col: int, amount: int = 1) -> None:
        if 0 <= channel < self._board.shape[0] and 0 <= row < self._board.shape[1] and 0 <= col < self._board.shape[2]:
            self._board[channel, row, col] += amount

    def _apply_actions(self, actions: Actions) -> None:
        """
        Apply each agent's chosen action to the environment.

        STUB: Currently only handles movement. Add collision detection,
        attack resolution, object interaction, etc. here.
        """
        changeFlagR = np.zeros(4)
        changeFlagB = np.zeros(4)
        for agent_id, action_num in actions.items():
            action = action_num % 9
            action_num //= 9
            pick = action_num % 6
            action_num //= 6
            drop = action_num % 6
            action_num //= 6
            score = action_num % 6
            if score==5:
                score==-1
            if drop==5:
                drop==-1
            if pick==5:
                pick==-1

            agent = self._agents[agent_id]
            if action == Action.STAY:
                if (pick ==-1 and drop ==-1 and score ==-1):
                    flags = self.setFlags(agent.row, agent.col)
                    if agent.team ==0:
                        changeFlagR[flags] = 1
                    if agent.team ==1:
                        changeFlagB[flags] = 1
                        

                if(pick==4 and agent.held_items[4]==1):
                    pick = -1
                if((pick==0 or pick==1 or pick==2 or pick==3) and (agent.held_items[0]==1 or agent.held_items[1]==1 or agent.held_items[2]==1 or agent.held_items[3]==1)):
                    pick = -1
                if pick!=-1:
                    if self._safe_board_value(pick, agent.row, agent.col) > 0:
                        self._safe_board_add(pick, agent.row, agent.col, -1)
                    elif self._safe_board_value(pick, agent.row - 1, agent.col) > 0:
                        self._safe_board_add(pick, agent.row - 1, agent.col, -1)
                    elif self._safe_board_value(pick, agent.row, agent.col - 1) > 0:
                        self._safe_board_add(pick, agent.row, agent.col - 1, -1)
                    elif self._safe_board_value(pick, agent.row + 1, agent.col) > 0:
                        self._safe_board_add(pick, agent.row + 1, agent.col, -1)
                    elif self._safe_board_value(pick, agent.row, agent.col + 1) > 0:
                        self._safe_board_add(pick, agent.row, agent.col + 1, -1)
                    elif self._safe_board_value(pick, agent.row + 1, agent.col + 1) > 0:
                        self._safe_board_add(pick, agent.row + 1, agent.col + 1, -1)
                    elif self._safe_board_value(pick, agent.row - 1, agent.col + 1) > 0:
                        self._safe_board_add(pick, agent.row - 1, agent.col + 1, -1)
                    elif self._safe_board_value(pick, agent.row + 1, agent.col - 1) > 0:
                        self._safe_board_add(pick, agent.row + 1, agent.col - 1, -1)
                    elif self._safe_board_value(pick, agent.row - 1, agent.col - 1) > 0:
                        self._safe_board_add(pick, agent.row - 1, agent.col - 1, -1)
                    else:
                        if(self.insideZone(agent.team, agent.row, agent.col)==1):
                            if agent.team==0 and self.redMatchload[pick]>0:
                                self.redMatchload[pick]-=1
                            elif agent.team==1 and self.blueMatchload[pick]>0:
                                self.blueMatchload[pick]-=1                            
                            else:
                                pick = -1
                        else:
                            pick = -1
                if pick!=-1:
                    agent.held_items[pick]+=1
                
                if(not self.checkScoring(agent.row, agent.col, agent.team)):
                    score = -1
                botSide = agent.scoreGoal(score)
                if(botSide!=-1): 
                    val = self.findGoal(agent.row, agent.col)
                    self.processGoal(val, score, botSide)

        for i in range (4):
            if(changeFlagR[i]==1 and changeFlagB[i]==0):
                self.zoneColors[i] = 1;
            if(changeFlagR[i]==0 and changeFlagB[i]==1):
                self.zoneColors[i] = 2;


        
        for agent_id, action_num in actions.items():
            action = action_num % 9
            action_num //= 9
            pick = action_num % 6
            action_num //= 6
            drop = action_num % 6
            action_num //= 6
            score = action_num % 6
            if score==5:
                score==-1
            if drop==5:
                drop==-1
            if pick==5:
                pick==-1
            
            agent = self._agents[agent_id]
            row = agent.row
            col = agent.col
            if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.U_L, Action.U_R, Action.D_L, Action.D_R):
                if action == Action.UP:
                    row = max(0, row - 1)
                elif action == Action.DOWN:
                    row = min(self._board_size - 1, row + 1)
                elif action == Action.LEFT:
                    col = max(0, col - 1)
                elif action == Action.RIGHT:
                    col = min(self._board_size - 1, col + 1)
                elif action == Action.U_L:
                    row = max(0, row - 1)
                    col = max(0, col - 1)
                elif action == Action.U_R:
                    row = max(0, row - 1)
                    col = min(self._board_size - 1, col + 1)
                elif action == Action.D_L:
                    row = min(self._board_size - 1, row + 1)
                    col = max(0, col - 1)
                elif action == Action.D_R:
                    row = min(self._board_size - 1, row + 1)
                    col = min(self._board_size - 1, col + 1)
            for agent_id2, val in self._agents.items():
                agent2 = self._agents[agent_id2]
                if(agent_id2 == agent_id):
                    continue
                if(self.checkCollision(row, col, agent2.row, agent2.col, 2*self.AGENT_RADIUS)):
                    self._agents[agent_id].stuck = True
            
            if(not self.checkCollisionObj(row, col)):
                self._agents[agent_id].stuck = True
            dropped, other_value = agent.move(action, drop, -1, self._board_size)
            if dropped>=0:
                self._safe_board_add(drop, agent.row, agent.col, 1)
            
                    
    def setFlags(row:int, col:int):
        if(col>=32 and row>=14 and row<=20):
            return 0
        if(row>=32 and col>=14 and col<=20):
            return 1
        if(col<=2 and row>=14 and row<=20):
            return 2
        if(row<=2 and col>=14 and col<=20):
            return 3
                    

    def insideZone(self, team:int, row:int, col:int)->int:
        if(team==0):
            if col<=5 or col>=29:
                if(row<=3):
                    return 1
                if(row>=31):
                    return -1
            return 0
        elif(team==1):
            if col<=5 or col>=29:
                if(row>=31):
                    return 1
                if(row<=3):
                    return -1
            return 0

    #team 0 = red

    def processGoal(self, val:int, score:int, bot:int):
        if(self.goalStates[val][3]<3 and bot<3):
            return
        if(self.goalStates[val][3]>2 and bot>2):
            return
        # print("processGoal "+str(self.goalStates[val]))
        if(self.goalStates[val][3]==3):
            self.goalStates[val][score]+=1
        if(self.goalStates[val][3]==4):
            self.goalStates[val][score]+=1
            self.goalStates[val][bot]+=1
        if(bot==3):
            self.goalStates[val][self.goalStates[val][3]]-=1
        self.goalStates[val][3] = score
        # print("processGoal "+str(self.goalStates[val]))
        


    def findGoal(self, x:int, y:int) -> int:
        for i in range(4, len(self.fieldObject)):
            if(self.checkCollision(
                x, y, self.fieldObject[i][0],
                self.fieldObject[i][1], 
                self.fieldObject[i][2]+self.AGENT_RADIUS+1
            )):
                return i
            

    def checkScoring(self, row:int , col:int , team:int) -> bool:
        if team==0:
            if(self.checkCollision(
                    row, col, self.fieldObject[0][0],
                    self.fieldObject[0][1], 
                    self.fieldObject[0][2]+self.AGENT_RADIUS+1
                )):
                return True
            if(self.checkCollision(
                    row, col, self.fieldObject[1][0],
                    self.fieldObject[1][1], 
                    self.fieldObject[1][2]+self.AGENT_RADIUS+1
                )):
                return True
        
        if team==1:
            if(self.checkCollision(
                    row, col, self.fieldObject[2][0],
                    self.fieldObject[2][1], 
                    self.fieldObject[2][2]+self.AGENT_RADIUS+1
                )):
                return True
            if(self.checkCollision(
                    row, col, self.fieldObject[3][0],
                    self.fieldObject[3][1], 
                    self.fieldObject[3][2]+self.AGENT_RADIUS+1
                )):
                return True
        
        for i in range(4, len(self.fieldObject)):
            if(self.checkCollision(
                row, col, self.fieldObject[i][0],
                self.fieldObject[i][1], 
                self.fieldObject[i][2]+self.AGENT_RADIUS+1
            )):
                return True
        return False

    def checkCollisionObj(self, row, col) -> bool:
        if row - self.AGENT_RADIUS + 1 < 0 or row + self.AGENT_RADIUS - 1 >= self._board_size:
            return False
        if col - self.AGENT_RADIUS + 1 < 0 or col + self.AGENT_RADIUS - 1 >= self._board_size:
            return False
        for i in range(len(self.fieldObject)):
            if(self.checkCollision(row, col, self.fieldObject[i][0],self.fieldObject[i][1], self.fieldObject[i][2]+self.AGENT_RADIUS)):
                return False
        return True
            


    def checkCollision(self, row1, col1, row2, col2, radius) -> bool:
        dist = np.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)
        return dist < radius

    def calcGoalScore(self)-> tuple[int, int]:
        blue = 0
        red = 0
        for  i in range(9):
            red+=self.goalStates[i][1]
            blue+=self.goalStates[i][2]
        if(self.zoneColors[0]==1):
            red += (self.goalStates[8][0]*10)
            red += (self.goalStates[2][0]*10)
        if(self.zoneColors[0]==2):
            blue += (self.goalStates[8][0]*10)
            blue += (self.goalStates[2][0]*10)

        if(self.zoneColors[1]==1):
            red += (self.goalStates[3][0]*10)
            red += (self.goalStates[6][0]*10)
        if(self.zoneColors[1]==2):
            blue += (self.goalStates[3][0]*10)
            blue += (self.goalStates[6][0]*10)

        if(self.zoneColors[2]==1):
            red += (self.goalStates[1][0]*10)
            red += (self.goalStates[5][0]*10)
        if(self.zoneColors[2]==2):
            blue += (self.goalStates[1][0]*10)
            blue += (self.goalStates[5][0]*10)

        if(self.zoneColors[3]==1):
            red += (self.goalStates[0][0]*10)
            red += (self.goalStates[7][0]*10)
        if(self.zoneColors[3]==2):
            blue += (self.goalStates[0][0]*10)
            blue += (self.goalStates[7][0]*10)
        return [red, blue]

    def calcMidScore(self)-> tuple[int, int]:
        blue = 0
        red = 0
        agent = self._agents["B0"]
        sum = abs(agent.row-17)
        sum+=abs(agent.col-17)
        if(sum<=8):
            red+=8

        agent = self._agents["B1"]
        sum = abs(agent.row-17)
        sum+=abs(agent.col-17)
        if(sum<=8):
            red+=8

        agent = self._agents["A0"]
        sum = abs(agent.row-17)
        sum+=abs(agent.col-17)
        if(sum<=8):
            blue+=8

        agent = self._agents["A1"]
        sum = abs(agent.row-17)
        sum+=abs(agent.col-17)
        if(sum<=8):
            blue+=8

        return [red, blue]



    def _compute_rewards(self, actions: Actions, red:int, blue:int, done:bool) -> Rewards:
        """
        Compute scalar reward for each agent after this step.

        STUB: Returns 0 for everyone. Replace with your game's reward signal:
            - Team wins/loses → +1 / -1
            - Capture objective → +0.5
            - Illegal move → -0.1
            - Time penalty → -0.01 per step
        """
        rewards: Rewards = {aid: 0.0 for aid in self.AGENT_IDS}
        red1, blue1 = self.calcGoalScore()
        rDiff = red1-red
        bDiff = blue1-blue
        red2, blue2 = self.calcMidScore()
        red1+=red2
        blue1+=blue2 
        if(done):
            if(red1>blue1):
                rewards["B0"]+=1
                rewards["B1"]+=1
                rewards["A0"]-=1
                rewards["A1"]-=1
            elif(red1<blue1):
                rewards["A0"]+=1
                rewards["A1"]+=1
                rewards["B0"]-=1
                rewards["B1"]-=1

        rewards["B0"]+=(rDiff*0.005-bDiff*0.005)
        rewards["B1"]+=(rDiff*0.005-bDiff*0.005)
        rewards["A0"]+=(bDiff*0.005-rDiff*0.005)
        rewards["A1"]+=(bDiff*0.005-rDiff*0.005)

        for aid in self.AGENT_IDS:
            agent = self._agents[aid]
            for i in range(len(agent.held_items)):
                if agent.held_items[i]!=0:
                    rewards[aid]+=0.0005
        # Example time-penalty to encourage faster resolution:
        for aid in self.AGENT_IDS:
            rewards[aid] -= 0.0006
        
        # ← Add your reward logic here           

        return rewards

    def _check_done(self) -> bool:
        """
        Return True when the episode should end.

        STUB: Ends only when STEPS_PER_EPISODE is reached. Add win/loss
        conditions (e.g., all opponents eliminated, objective captured).
        """
        if self._step_count >= STEPS_PER_EPISODE:
            return True

        # ← Add win/loss conditions here
        return False
    

    def _rebuild_board(self) -> None:
        """Recompute the board tensor from current agent positions."""
        # self._board[:] = 0.0
        # self._board[0] = 1.0  # Channel 0: everything is "empty" by default

        # for agent in self._agents.values():
        #     if not agent.alive:
        #         continue
        #     channel = 1 if agent.team == TEAM_A else 2
        #     self._board[channel, agent.row, agent.col] = 1.0
        #     self._board[0, agent.row, agent.col] = 0.0  # no longer empty*/
