"""
Marcus' little prototype for a Monte Carlo Tree Search model, based on self-taken notes and interpretation.
"""

import math

RUNTIME = 10

"""
Class for nodes, the basic building blocks of the model.
"""
class Node:
    def __init__():
        cost = 0


"""
This function begins the search, growing the tree node-by-node.
"""
def search():

    # These are placeholder variables, these will likely be assigned to each node or something.
    state_value = 0
    state_visits = 0
    actions_in_state = 0

    # Loop here is for taking actions during the compute time and building the tree.
    for i in range(RUNTIME):
        ucb = state_value - math.sqrt(math.log(state_visits) / actions_in_state)
        new_node = Node()