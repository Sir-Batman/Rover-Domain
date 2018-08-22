"""
Module to calculate alignment between two rewards.
"""

def alignment(reward_a, reward_b, state):
    """
    Calculates the alignment along 8 actions, assuming the actions
    correspond to movements in the real world.
    Requires a method to create different states, and a the two reward functions to compare.

    Args:
        reward_a: The first reward function to test
        reward_b: The second reward function to test
        state: The current state of the agent

    Returns:
        Float: the alignment as calculated by sampling the 8 different actions.
    """
    actions = [(0, 1), (1,0), (0,-1), (-1,0), (0.71, 0.71), (0.71, -0.71), (-0.71, 0.71), (-0.71, -0.71)]

    alignment = 0
    for a in actions:
        # apply action onto agent 
        # Evaluate rewards
        alignment += (reward_a(state) - reward_a(state_prime))*(reward_b(state) - reward_b(state_prime))
    alignment /= 8
    return alignment
