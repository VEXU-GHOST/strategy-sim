"""
Marcus' little prototype for a Monte Carlo Tree Search model, based on self-taken notes and interpretation.
"""


# The math library is used for advanced calculations done in evaluation.
# The random library is used for generating random numbers for robot states.
import math
import random


# The runtime here determines how long the main loop will run.
RUNTIME = 4


"""
Class for nodes, the basic building blocks of the model.
"""
class Node:

    # The node's attributes are initialized here.
        # The state holds an x and y for position, and a angle for rotation.
        # The parent is used for a backtracking pointer toward the prior node with the prior state.
        # A list of children nodes holds the posterior nodes and their states.
        # This is the integer value for the amount of visits the node received.
        # The reward will be measured, somehow...
    def __init__(self):
        self.state = None
        self.parent = None
        self.children = []
        self.visits = 0
        self.reward = 0.0


"""
Function for generating a random action to attach to a node, either a rotation or a translation.
"""
def new_state(state):

    # A new state copies the variables from the old state taken in as a parameter.
    # If the flag is zero, the robot will translate.
        # This sets a random x-coordinate, directional dynamics are not implemented yet.
        # This sets a random y-coordinate, directional dynamics are not implemented yet.
    # If the flag is one, the robot will rotate.
        # An angle will be added onto the robot's original state.
    # The modified, new state is returned.
    new_state = state.copy()
    if random.randint(0, 1) == 0:
        new_state[0] = random.randint(-4, 4)
        new_state[1] = random.randint(-4, 4)
    else:
        new_state[2] += 45 * random.randint(-4, 4)
    return new_state


"""
This function begins the search, growing the tree node-by-node.
"""
def search():

    # The state value measures how likely the state is to be tried.
    # The total number of times a state is visited.
    # The total number of actions taken while in the state.
    state_value = 0
    state_visits = 0
    actions_in_state = 0

    # The state is initialized here, x, y, and angle.
    # A node object is initialized, this will be the root.
    # That object's state variable is set to the initial.
    # The root's cost is zero, naturally.
    state = [0, 0, 0]
    node = Node()
    node.state = state
    node.cost = 0

    actions_to_take = []
    actions_to_take.append(node)

    # There should likely be another for loop here based on a list containing all states to try in a run.
    for i in range(RUNTIME):

        # A list here initializes the new actions to take, reset every time 
        new_actions = []
        
        # Loop here is for taking actions during the compute time and building the tree.
        for node in actions_to_take:

            # This loop generates two nodes that branch off of a parent.
                # The next node generated as an object is the child.
                # The child node's parent is set to the node before.
                # The next state set calculated by a function.
                # The child node's state is set.
                # The child is added to the parent's list of children.
            for i in range(2): # Please replace "2" with "random.randint(1, 5)"" when ready.
                next_state = []
                next_node = Node()
                next_node.parent = node
                next_state = new_state(node.state)
                next_node.state = next_state
                node.children.append(next_node)
                new_actions.append(next_node)

        # Here is a debugging print statement that prints the nodes in the actions taken, and their children.
        for node in actions_to_take:
            print(node.state, [child.state for child in node.children])

        # The list for actions to take is cleared, with the new actions replacing, and should double the length of the action loop.
        actions_to_take = new_actions


    # This line calculates the upper confidence bound.
    # ucb = state_value - math.sqrt(math.log(state_visits) / actions_in_state)