import random
import torch
import src.ccea


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
    """ Returns a function which makes allows for using the teaming 2 policy """
    # Pick one of the team 2 policies
    filepath = "policies/Team2Policy{}.model".format(0)
    model = src.ccea.Evo_MLP(8, 2)
    model.load_state_dict(torch.load(filepath))
    return lambda s: model.get_next(s)


def team_3_policy():
    """ Returns a function which makes allows for using the teaming 3 policy """
    # Pick one of the team 3 policies
    filepath = "policies/Team3Policy{}.model".format(0)
    model = src.ccea.Evo_MLP(8, 2)
    model.load_state_dict(torch.load(filepath))
    return lambda s: model.get_next(s)


def team_4_policy():
    """ Returns a function which makes allows for using the teaming 4 policy """
    # Pick one of the team 4 policies
    filepath = "policies/Team4Policy{}.model".format(0)
    model = src.ccea.Evo_MLP(8, 2)
    model.load_state_dict(torch.load(filepath))
    return lambda s: model.get_next(s)


def random_policy(state):
    return (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
