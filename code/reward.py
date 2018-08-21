from numpy import dot

def calculateGlobalReward(data):
    """
    Calculates and returns the global reward for the current state of the world, as
    input by data.

    Args:
        data: the global data structure of the simulation

    Returns:
        float: G for the entire team, based on the path performance over the episode.

    Preconditions:
        Data contains the agent history, POI locations, and task metadata necessary
        to calculate G.

    Postconditions:
        No modifications are made to data, reward value is returned.
    """
    agentPositionHistory = data["Agent Position History"]
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    poiPositionCol = data["Poi Positions"]
    minDistanceSqr = data["Minimum Distance"] ** 2
    historyStepCount = data["Steps"] + 1
    coupling = data["Coupling"]
    observationRadiusSqr = data["Observation Radius"] ** 2
    
    globalReward = 0.0
    
    for poiIndex in range(number_pois):
        poiPosition = poiPositionCol[poiIndex]
        closestObsDistanceSqr = float("inf")
        for stepIndex in range(historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = float("inf")
            for agentIndex in range(number_agents):
                # Calculate separation distance between poi and agent
                agentPosition = agentPositionHistory[agentIndex][stepIndex]
                separation = poiPosition - agentPosition
                distanceSqr = dot(separation, separation)
                # Check if agent observes poi, update closest step distance
                if distanceSqr < observationRadiusSqr:
                    observerCount += 1
                    if distanceSqr < stepClosestObsDistanceSqr:
                        stepClosestObsDistanceSqr = distanceSqr
            # update closest distance only if poi is observed    
            if observerCount >= coupling:
                if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                    closestObsDistanceSqr = stepClosestObsDistanceSqr
        # add to global reward if poi is observed 
        if closestObsDistanceSqr < observationRadiusSqr:
            if closestObsDistanceSqr < minDistanceSqr:
                closestObsDistanceSqr = minDistanceSqr
            globalReward += 1.0 / closestObsDistanceSqr

    return globalReward

def assignGlobalReward(data):
    """
    Calculates G and assigns values into the global data structure.
    See calculateGlobalReward() for implementation specifics.

    Args:
        data: global data structure

    Postconditions:
        data is updated with the "Global Reward" field
        data is updated with the "Agent Rewards" field being assigned G for each agent.

    """
    globalReward = calculateGlobalReward(data)
    data["Global Reward"] = globalReward
    data["Agent Rewards"] = [globalReward] * number_agents
 
def assignStepGlobalReward(data):
    agentPositionHistory = data["Agent Position History"]
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    poiPositionCol = data["Poi Positions"]
    minDistanceSqr = data["Minimum Distance"] ** 2
    historyStepCount = data["Steps"] + 1
    coupling = data["Coupling"]
    observationRadiusSqr = data["Observation Radius"] ** 2
    
    globalReward = 0.0
    
    for poiIndex in range(number_pois):
        poiPosition = poiPositionCol[poiIndex]
        for stepIndex in range(historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = float("inf")
            for agentIndex in range(number_agents):
                # Calculate separation distance between poi and agent
                agentPosition = agentPositionHistory[agentIndex][stepIndex]
                separation = poiPosition - agentPosition
                distanceSqr = dot(separation, separation)
                
                # Check if agent observes poi, update closest step distance
                if distanceSqr < observationRadiusSqr:
                    observerCount += 1
                    if distanceSqr < stepClosestObsDistanceSqr:
                        stepClosestObsDistanceSqr = distanceSqr
                        
                        
            # update closest distance only if poi is observed    
            if observerCount >= coupling:
                # add to global reward if poi is observed 
                if stepClosestObsDistanceSqr < observationRadiusSqr:
                    if stepClosestObsDistanceSqr < minDistanceSqr:
                        stepClosestObsDistanceSqr = minDistanceSqr
                    globalReward += 1.0 / stepClosestObsDistanceSqr

    data["Global Reward"] = globalReward
    data["Agent Rewards"] = [globalReward] * number_agents    
    

def calculateDifferenceReward(data):
    """
    Calculates the path-based difference reward for all agents in the team. 

    Args:
        data: global data structure for the simulation

    Returns:
        Tuple: ([Agent Difference Rewards], Global Reward)

    Preconditions: 
        data is well constructed to pull the agent history, POI locations, and task metadata
        needed to calculate G & D
    """
    agentPositionHistory = data["Agent Position History"]
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    poiPositionCol = data["Poi Positions"]
    minDistanceSqr = data["Minimum Distance"] ** 2
    historyStepCount = data["Steps"] + 1
    coupling = data["Coupling"]
    observationRadiusSqr = data["Observation Radius"] ** 2

    # Calculate the global reward 
    globalReward = calculateGlobalReward(data)

    # Start calculating D for each agent
    differenceRewards = [0] * number_agents

    for agentIndex in range(number_agents):
        globalWithoutReward = 0
        for poiIndex in range(number_pois):
            poiPosition = poiPositionCol[poiIndex]
            closestObsDistanceSqr = float("inf")
            for stepIndex in range(historyStepCount):
                # Count how many agents observe poi, update closest distance if necessary
                observerCount = 0
                stepClosestObsDistanceSqr = float("inf")
                for otherAgentIndex in range(number_agents):
                    if agentIndex != otherAgentIndex:
                        # Calculate separation distance between poi and agent
                        agentPosition = agentPositionHistory[otherAgentIndex][stepIndex]
                        separation = poiPosition - agentPosition
                        distanceSqr = dot(separation, separation)
                        
                        # Check if agent observes poi, update closest step distance
                        if distanceSqr < observationRadiusSqr:
                            observerCount += 1
                            if distanceSqr < stepClosestObsDistanceSqr:
                                stepClosestObsDistanceSqr = distanceSqr
                            
                # update closest distance only if poi is observed    
                if observerCount >= coupling:
                    if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                        closestObsDistanceSqr = stepClosestObsDistanceSqr
            
            # add to global reward if poi is observed 
            if closestObsDistanceSqr < observationRadiusSqr:
                if closestObsDistanceSqr < minDistanceSqr:
                    closestObsDistanceSqr = minDistanceSqr
                globalWithoutReward += 1.0 / closestObsDistanceSqr
        differenceRewards[agentIndex] = globalReward - globalWithoutReward

    # Return D values and G, in case we need G later on so we don't need to recalculate it
    return differenceRewards, globalReward

def assignDifferenceReward(data):
    """
    Calculates D and assigns values into the global data structure.
    See calculateDifferenceReward() for implementation specifics.

    Args:
        data: global data structure

    Postconditions:
        data is updated with the "Global Reward" field
        data is updated with the "Agent Rewards" field being assigned D for each agent.

    """
    differenceRewards, globalReward = calculateDifferenceReward(data)
    data["Agent Rewards"] = differenceRewards  
    data["Global Reward"] = globalReward
