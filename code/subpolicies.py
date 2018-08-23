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
