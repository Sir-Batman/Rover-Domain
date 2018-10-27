from alignment import quickalignment
import numpy as np
import subpolicies
import qlearner
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


def doAgentProcess_ScheduleMP(data):
    """
    Uses data['Policy Schedule'] to determine which policy of the set of policies
    each agent should be using at each time step.
    
    Each Schedule is structured as 
    [(policy, timestep), ... , (p, t)]
    where p is the index/descriptor to use for a policy, and
    t is the timestep at which to switch to policy p (not the duration of using policy p)
    The first policy to be used should have timestep 0 as it's start point.

    Also assumes the agent policies are stored in a list
    """
    # Find policy to use
    current_step = data["Step Index"]
    policy_key = None
    for i, (p, t) in enumerate(data['Policy Schedule']):
        if t > current_step:
            policy_key = data['Policy Schedule'][i-1][0]
            break
    # Edge case for the last policy in the schedule
    if policy_key is None:
        policy_key = data['Policy Schedule'][-1][0]

    # Start the action selection based on right policy from above
    cdef int agentIndex
    cdef int number_agents = data['Number of Agents']
    actionCol = np.zeros((number_agents, 2), dtype = np.float_)
    observationCol = data["Agent Observations"]
    for agentIndex in range(number_agents):
        policy = data["Agent Policies"][agentIndex][policy_key]
        actionCol[agentIndex][0], actionCol[agentIndex][1] = policy(observationCol[agentIndex])
    data["Agent Actions"] = actionCol


def doAgentProcess_Alignment(data):
    """
    Inserts the alignment calculation and policy-action selection into the agent process.
    Uses subpolicies imported from subpolicies.py
    """
    cdef int agentIndex
    cdef int number_agents = data['Number of Agents']
    actionCol = np.zeros((number_agents, 2), dtype = np.float_)
    observationCol = data["Agent Observations"]
    for agentIndex in range(number_agents):
        # Apply the alignment calculation and action selection for each agent
        aligned_reward = quickalignment(data, agentIndex)
        if aligned_reward == "agent":
            # Go to agent is most aligned
            actionCol[agentIndex][0], actionCol[agentIndex][1] = subpolicies.agent_policy(observationCol[agentIndex])
        else:
            # POI most aligned
            actionCol[agentIndex][0], actionCol[agentIndex][1] = subpolicies.poi_policy(observationCol[agentIndex])
    data["Agent Actions"] = actionCol

def to_Discrete(state):
    """
    Translate the current, continuous valued vector state into a discrete version.
    This is done by scaling up the continuous values by a factor of 10, then rounding
    to the closest integer.

    Args:
        state - the 8-dimensional continuous value vector state in the rover domain

    Returns:
        state: numpy array of integers. Will be the same length (8) as the input vector.

    Preconditions: None

    Postconditions: None
    """
    new_state = np.zeros(state.shape())
    for i, value in enumerate(state):
        new_state[i] = round(value*10)
    return new_state

def doAgentProcess_Q_agent(data):
    """
    Processes the observations/decisions for the Q-Agent.
    This version has the Q-Agent try to learn a which action to take based on the modified state representation.
    TODO: make this agent learn which sub-policy to select from. Will not use alignment, but will be a basic HRL model.

    Args:
        data: the global data structure

    Returns: None

    Preconditions:
        data['Agent Policies'] has the Q-Agent class instantiations

    Postconditions:
        data['Agent Actions'] is populated with the actions selected by each agent.
    """
    action_map = {0:[1,0], 1:[1.4, 1.4], 2:[0,1],  3:[-1.4, 1.4], 4:[-1, 0], 5:[-1.4, -1.4], 6:[0,-1], 7:[1.4, -1.4]}
    actions = np.zeros((data['Number of Agents'], 2))
    for agent_index in range(data['Number of Agents']):
        discrete_state = to_Discrete(data['Agent Observations'][agent_index])
        selected_action = data['Agent Policies'][agent_index](discrete_state)
        actions[agent_index] = action_map[selected_action] # translate from an action number to the vector
    data["Agent Actions"] = actions

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
