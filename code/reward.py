from numpy import dot

def assignGlobalReward(data):
    agentPositionHistory = data["Agent Position History"]
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    poiValueCol = data['Poi Values']
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
            globalReward += poiValueCol[poiIndex] / closestObsDistanceSqr
    
    data["Global Reward"] = globalReward
    data["Agent Rewards"] = [globalReward] * number_agents
 
def assignStepGlobalReward(data):
    agentPositionHistory = data["Agent Position History"]
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    poiValueCol = data['Poi Values']
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
                    globalReward += poiValueCol[poiIndex] / stepClosestObsDistanceSqr

    data["Global Reward"] = globalReward
    data["Agent Rewards"] = [globalReward] * number_agents    
    
def assignDifferenceReward(data):
    agentPositionHistory = data["Agent Position History"]
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    poiValueCol = data['Poi Values']
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
            globalReward += poiValueCol[poiIndex] / closestObsDistanceSqr
    
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
                globalWithoutReward += poiValueCol[poiIndex] / closestObsDistanceSqr
        differenceRewards[agentIndex] = globalReward - globalWithoutReward
        
    data["Agent Rewards"] = differenceRewards  
    data["Global Reward"] = globalReward

            
    
    