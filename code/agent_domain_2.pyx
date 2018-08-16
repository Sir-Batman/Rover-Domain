
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double m)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def doAgentSense(data):
    """
     Sensor model is <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
     Where a means (other) agent, p means poi, and the rest are the quadrants
    """
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    
    agentPositionCol = np.array(data["Agent Positions"])
    cdef double[:, :] agentPositionColView = agentPositionCol
    
    poiPositionCol = np.array(data["Poi Positions"])
    cdef double[:, :] poiPositionColView = poiPositionCol
    
    orientationCol = np.array(data["Agent Orientations"])
    cdef double[:, :] orientationColView = orientationCol
    
    observationCol = np.zeros((number_agents, 8), dtype = np.float64)
    cdef double[:, :] observationColView = observationCol
    
    cdef int agentIndex, otherAgentIndex, poiIndex, obsIndex



    cdef double globalFrameSeparation0, globalFrameSeparation1
    cdef double agentFrameSeparation0, agentFrameSeparation1

    cdef double distanceSqr
    
    
    for agentIndex in range(number_agents):

        # calculate observation values due to other agents
        for otherAgentIndex in range(number_agents):
            
            # agents do not sense self (ergo skip self comparison)
            if agentIndex == otherAgentIndex:
                continue
                
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = agentPositionColView[otherAgentIndex,0] - agentPositionColView[agentIndex,0]
            globalFrameSeparation1 = agentPositionColView[otherAgentIndex,1] - agentPositionColView[agentIndex,1]
            
            # Translate separation to agent frame using inverse rotation matrix
            agentFrameSeparation0 = orientationColView[agentIndex, 0] * globalFrameSeparation0 + orientationColView[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationColView[agentIndex, 0] * globalFrameSeparation1 - orientationColView[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
        
            
            # other is east of agent
            if agentFrameSeparation0 > 0:
                # other is north-east of agent
                if agentFrameSeparation1 > 0:
                    observationColView[agentIndex,0] += 1.0 / distanceSqr
                else: # other is south-east of agent
                    observationColView[agentIndex,3] += 1.0  / distanceSqr
            else:  # other is west of agent
                # other is north-west of agent
                if agentFrameSeparation1 > 0:
                    observationColView[agentIndex,1] += 1.0  / distanceSqr
                else:  # other is south-west of agent
                    observationColView[agentIndex,2] += 1.0  / distanceSqr



        # calculate observation values due to pois
        for poiIndex in range(number_pois):
            
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = poiPositionColView[poiIndex,0] - agentPositionColView[agentIndex,0]
            globalFrameSeparation1 = poiPositionColView[poiIndex,1] - agentPositionColView[agentIndex,1]
            
            # Translate separation to agent frame unp.sing inverse rotation matrix
            agentFrameSeparation0 = orientationColView[agentIndex, 0] * globalFrameSeparation0 + orientationColView[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationColView[agentIndex, 0] * globalFrameSeparation1 - orientationColView[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
            
            # poi is east of agent
            if agentFrameSeparation0> 0:
                # poi is north-east of agent
                if agentFrameSeparation1 > 0:
                    observationColView[agentIndex,4] += 1.0  / distanceSqr
                else: # poi is south-east of agent
                    observationColView[agentIndex,7] += 1.0  / distanceSqr
            else:  # poi is west of agent
                # poi is north-west of agent
                if agentFrameSeparation1 > 0:
                    observationColView[agentIndex,5] += 1.0  / distanceSqr
                else:  # poi is south-west of agent
                    observationColView[agentIndex,6] += 1.0  / distanceSqr
                    
    data["Agent Observations"] = observationCol

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
def doAgentProcess(data):
    cdef int number_agents = data['Number of Agents']
    actionCol = np.zeros((number_agents, 2), dtype = np.float_)
    policyCol = data["Agent Policies"]
    policyFuncCol = list(map(lambda x: x.get_next, policyCol))
    observationCol = data["Agent Observations"]
    cdef int agentIndex
    for agentIndex in range(number_agents):
        actionCol[agentIndex] = policyFuncCol[agentIndex](observationCol[agentIndex])
    data["Agent Actions"] = actionCol

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.  
def doAgentMove(data):
    cdef float worldWidth = data["World Width"]
    cdef float worldLength = data["World Length"]
    cdef int number_agents = data['Number of Agents']
    
    agentPositionCol = np.array(data["Agent Positions"])
    cdef double[:, :] agentPositionColView = agentPositionCol
    
    orientationCol = np.array(data["Agent Orientations"])
    cdef double[:, :] orientationColView = orientationCol
    
    actionCol = np.array(data["Agent Actions"]).astype(np.float_)
    cdef double[:, :] actionColView = actionCol
    
    cdef int agentIndex

    cdef double globalFrameMotion0, globalFrameMotion1, norm
    
    # move all agents
    for agentIndex in range(number_agents):

        # turn action into global frame motion
        globalFrameMotion0 = orientationColView[agentIndex, 0] * actionColView[agentIndex, 0] - orientationColView[agentIndex, 1] * actionColView[agentIndex, 1] 
        globalFrameMotion1 = orientationColView[agentIndex, 0] * actionColView[agentIndex, 1] + orientationColView[agentIndex, 1] * actionColView[agentIndex, 0] 
        
      
        # globally move and reorient agent
        agentPositionColView[agentIndex, 0] += globalFrameMotion0
        agentPositionColView[agentIndex, 1] += globalFrameMotion1
        
        if globalFrameMotion0 == 0.0 and globalFrameMotion1 == 0.0:
            orientationColView[agentIndex,0] = 1.0
            orientationColView[agentIndex,1] = 0.0
        else:
            norm = sqrt(globalFrameMotion0**2 +  globalFrameMotion1 **2)
            orientationColView[agentIndex,0] = globalFrameMotion0 /norm
            orientationColView[agentIndex,1] = globalFrameMotion1 /norm
            
        # Check if action moves agent within the world bounds
        if agentPositionColView[agentIndex,0] > worldWidth:
            agentPositionColView[agentIndex,0] = worldWidth
        elif agentPositionColView[agentIndex,0] < 0.0:
            agentPositionColView[agentIndex,0] = 0.0
        
        if agentPositionColView[agentIndex,1] > worldLength:
            agentPositionColView[agentIndex,1] = worldLength
        elif agentPositionColView[agentIndex,1] < 0.0:
            agentPositionColView[agentIndex,1] = 0.0
        

    data["Agent Positions"]  = agentPositionCol
    data["Agent Orientations"] = orientationCol 