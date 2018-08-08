from numpy import matmul, array, dot, zeros, linalg, sin

def doAgentSense(data):
    """
     Sensor model is <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
     Where a means (other) agent, p means poi, and the rest are the quadrants
    """
    
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    agentPositionCol = data["Agent Positions"]
    poiPositionCol = data["Poi Positions"]
    orientationCol = data["Agent Orientations"]
    minDistanceSqr = data["Minimum Distance"] ** 2
    observationCol = [None] * number_agents
    
    
    for agentIndex in range(number_agents):
        
        # recover agent position and orientation
        agentPosition = agentPositionCol[agentIndex]
        agentOrientation = orientationCol[agentIndex]
        c = agentOrientation[0]; s = agentOrientation[1]

        # initialize observation to zero
        observation = zeros(8)
        
        # calculate observation values due to other agents
        for otherAgentIndex in range(number_agents):
            
            # agents do not sense self (ergo skip self comparison)
            if agentIndex == otherAgentIndex:
                continue
                
            # Get global separation vector between the two agents    
            otherPosition = agentPositionCol[otherAgentIndex]
            globalFrameSeparation = otherPosition - agentPosition
            
            # Translate separation to agent frame using inverse rotation matrix
            agentFrameSeparation = matmul(array([[c, s], [-s, c]]), globalFrameSeparation)
            distanceSqr = dot(agentFrameSeparation, agentFrameSeparation)
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
            
            
            # other is east of agent
            if agentFrameSeparation[0] > 0:
                # other is north-east of agent
                if agentFrameSeparation[1] > 0:
                    observation[0] += 1.0 / distanceSqr
                else: # other is south-east of agent
                    observation[3] += 1.0  / distanceSqr
            else:  # other is west of agent
                # other is north-west of agent
                if agentFrameSeparation[1] > 0:
                    observation[1] += 1.0  / distanceSqr
                else:  # other is south-west of agent
                    observation[2] += 1.0  / distanceSqr

        # calculate observation values due to pois
        for poiIndex in range(number_pois):
            
            # Get global separation vector between the two agents    
            poiPosition = poiPositionCol[poiIndex]
            globalFrameSeparation = poiPosition - agentPosition
            
            # Translate separation to agent frame using inverse rotation matrix
            agentFrameSeparation = matmul(array([[c, s], [-s, c]]), globalFrameSeparation)
            distanceSqr = dot(agentFrameSeparation, agentFrameSeparation)
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
            
            
            # poi is east of agent
            if agentFrameSeparation[0] > 0:
                # poi is north-east of agent
                if agentFrameSeparation[1] > 0:
                    observation[4] += 1.0  / distanceSqr
                else: # poi is south-east of agent
                    observation[7] += 1.0  / distanceSqr
            else:  # poi is west of agent
                # poi is north-west of agent
                if agentFrameSeparation[1] > 0:
                    observation[5] += 1.0  / distanceSqr
                else:  # poi is south-west of agent
                    observation[6] += 1.0  / distanceSqr
            
        observationCol[agentIndex] = observation
    data["Agent Observations"] = observationCol
    
def doAgentProcess(data):
    number_agents = data['Number of Agents']
    actionCol = [None] * number_agents
    policyCol = data["Agent Policies"]
    observationCol = data["Agent Observations"]
    for agentIndex in range(number_agents):
        actionCol[agentIndex] = policyCol[agentIndex].get_next(observationCol[agentIndex])
    data["Agent Actions"] = actionCol
     
def doAgentMove(data):
    worldWidth = data["World Width"]
    worldLength = data["World Length"]
    number_agents = data['Number of Agents']
    actionCol = data["Agent Actions"]
    positionCol = data["Agent Positions"]
    orientationCol = data["Agent Orientations"]
    
    # move all agents
    for agentIndex in range(number_agents):
        # recover agent position and orientation
        agentPosition = positionCol[agentIndex]
        agentOrientation = orientationCol[agentIndex]
        c = orientationCol[agentIndex][0]; s = orientationCol[agentIndex][1]

        # turn action into global frame motion
        agentFrameMotion = actionCol[agentIndex]
        globalFrameMotion = matmul(array([[c, -s], [s, c]]), agentFrameMotion)
      
        # globally move and reorient agent
        positionCol[agentIndex] += globalFrameMotion
        if (globalFrameMotion == zeros(2)).all():
            orientationCol[agentIndex] = array([1.0,0.0])
        else:
            orientationCol[agentIndex] = globalFrameMotion / linalg.norm(globalFrameMotion)
            
        # Check if action moves agent within the world bounds
        if positionCol[agentIndex][0] > worldWidth:
            positionCol[agentIndex][0] = worldWidth
        elif positionCol[agentIndex][0] < 0.0:
            positionCol[agentIndex][0] = 0.0
        
        if positionCol[agentIndex][1] > worldLength:
            positionCol[agentIndex][1] = worldLength
        elif positionCol[agentIndex][1] < 0.0:
            positionCol[agentIndex][1] = 0.0
        
    data["Agent Positions"]  = positionCol
    data["Agent Orientation"] = orientationCol 