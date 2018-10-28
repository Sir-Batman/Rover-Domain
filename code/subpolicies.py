import code.ccea
import torch
import pickle
def poi_policy(state):
    """
    Heuristic method, drives the agent toward the closest POI by the state representation
    Sensor model is <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
    """
    strongest_poi = max(state[4:])
    if state[4] == strongest_poi:
        return (0.71, 0.71)
    elif state[5] == strongest_poi:
        return (-0.71, 0.71)
    elif state[6] == strongest_poi:
        return (-0.71, -0.71)
    elif state[7] == strongest_poi:
        return (0.71, -0.71)

def agent_policy(state):
    """
    Heuristic method, drives the agent toward the closest agent by the state representation
    Sensor model is <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
    """
    strongest_agent = max(state[0:4])
    if state[0] == strongest_agent:
        return (0.71, 0.71)
    elif state[1] == strongest_agent:
        return (-0.71, 0.71)
    elif state[2] == strongest_agent:
        return (-0.71, -0.71)
    elif state[3] == strongest_agent:
        return (0.71, -0.71)


def team_2_policy():
    # Pick one of the team 2 policies
    filepath = "policies/Team2Policy{}.model".format(0)
    model = code.ccea.Evo_MLP(8, 2)
    model.load_state_dict(torch.load(filepath))
    return lambda s: model.get_next(s)

"""
def team3(state):
    # Pick one of the team
    filepath = "policies/team2policy{}.model".format(0)
    policy = pickle.load("")
    return
"""


