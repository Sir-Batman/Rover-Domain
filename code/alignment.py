import numpy as np
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

def quickalignment(data, agentIndex):
    '''
    Does a quick and dirty alignment check for gotorover and gotopoi
    Sees which one increases more overall
    mathmatically same as full alignment but slightly easier to implement tonight
    '''
    POI_score = 0
    agent_score = 0
    agent_position = np.array(data["Agent Positions"])[agentIndex]
    # check 8 directions
    actions = [(0, 1), (1,0), (0,-1), (-1,0), (0.71, 0.71), (0.71, -0.71), (-0.71, 0.71), (-0.71, -0.71)]
    for a in actions:
        new_pos = agent_position + np.array(a)
        for POI in data["Poi Positions"]:
            if np.dot(POI - new_pos, POI - new_pos) < np.dot(POI - agent_position, POI - agent_position):
                # Got closer to this POI
                POI_score += np.dot(POI - new_pos, POI - new_pos)
        for agent in data["Agent Positions"]:
            if np.dot(agent - new_pos, agent - new_pos) < np.dot(agent - agent_position, agent - agent_position):
                # Got closer to agent
                agent_score += np.dot(agent - new_pos, agent - new_pos) 
    if agent_score > POI_score:
        return "agent"
    else:
        return "poi"
