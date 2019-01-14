import numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double m)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

cpdef assignGlobalRewardMod(data):
    
    cdef int[:] itemHeld=data["Item Held"]
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef int historyStepCount = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double observationRadiusSqr = data["Observation Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
  
    
    cdef int poiIndex, stepIndex, agentIndex, observerCount
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    cdef double globalReward = 0.0
 
    
    for poiIndex in range(number_pois//2):
        closestObsDistanceSqr = Inf
        for stepIndex in range(historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = Inf
            for agentIndex in range(number_agents):
                # Calculate separation distance between poi and agent
                separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                distanceSqr = separation0 * separation0 + separation1 * separation1
                
                # Check if agent observes poi, update closest step distance
                if distanceSqr < observationRadiusSqr and itemHeld[agentIndex]:
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
    data["Agent Rewards"] = np.ones(number_agents) * globalReward 
    
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef doAgentSenseMod(data):
    """
     Sensor model is <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
     Where a means (other) agent, p means poi, and the rest are the quadrants
    """
    cdef double obsRadius=data["Observation Radius"] ** 2
    cdef double viewDistance = data['View Distance'] ** 2
    
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef double[:, :] agentPositionCol = data["Agent Positions"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
    cdef double[:, :] orientationCol = data["Agent Orientations"]
    npObservationCol = np.zeros((number_agents, 8), dtype = np.float64)
    
    
   
    
    cdef int[:] itemHeld
    
    if data["Sequential"]:
        itemHeld=data["Item Held"]
        npObservationCol = np.zeros((number_agents, 13), dtype = np.float64)
    else:
        npObservationCol = np.zeros((number_agents, 8), dtype = np.float64)
    cdef double[:, :] observationCol = npObservationCol
    
    
    cdef int agentIndex, otherAgentIndex, poiIndex, obsIndex, shift
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
            globalFrameSeparation0 = agentPositionCol[otherAgentIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = agentPositionCol[otherAgentIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame using inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            if viewDistance > 0 and distanceSqr > viewDistance :
                continue
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
        	
            
            # other is east of agent
            if agentFrameSeparation0 > 0:
                # other is north-east of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,0] += 1.0 / distanceSqr
                else: # other is south-east of agent
                    observationCol[agentIndex,3] += 1.0  / distanceSqr
            else:  # other is west of agent
                # other is north-west of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,1] += 1.0  / distanceSqr
                else:  # other is south-west of agent
                    observationCol[agentIndex,2] += 1.0  / distanceSqr



        # calculate observation values due to pois
        for poiIndex in range(number_pois):
            
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = poiPositionCol[poiIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = poiPositionCol[poiIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame unp.sing inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            if viewDistance > 0 and distanceSqr > viewDistance:
                continue
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
            
            # half of poi give a "key" and the other half are a "lock" which need multiple agents to unlock
            shift=0
            
            if (poiIndex < number_pois//2 and data["Sequential"]):
                shift = 4
                
                if (obsRadius > distanceSqr):
                    itemHeld[agentIndex]=1
                
                
                if ( itemHeld[agentIndex] ) :
                    observationCol[12]=1
            
            # poi is east of agent
            if agentFrameSeparation0> 0:
                # poi is north-east of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,4+shift] += poiValueCol[poiIndex]  / distanceSqr
                else: # poi is south-east of agent
                    observationCol[agentIndex,7+shift] += poiValueCol[poiIndex]  / distanceSqr
            else:  # poi is west of agent
                # poi is north-west of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,5+shift] += poiValueCol[poiIndex]  / distanceSqr
                else:  # poi is south-west of agent
                    observationCol[agentIndex,6+shift] += poiValueCol[poiIndex]  / distanceSqr
                    
    data["Agent Observations"] = npObservationCol


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

cpdef assignGlobalRewardRecipe(data):
    
    cdef int[:,:] itemHeld=data["Item Held"]
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef int historyStepCount = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double observationRadiusSqr = data["Observation Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
  
    
    cdef int[:]  recipe = data["Recipe"]
    cdef int recipeSize = data["Recipe Size"]
    cdef int nPoiTypes  = data["Number of POI Types"]
    cdef int ordered    = data["Ordered"] 
    
    cdef int poiIndex, stepIndex, agentIndex, observerCount, poiType
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    
    cdef double globalReward = 0.0
 
    
    for poiIndex in range(number_pois):
        poiType = poiIndex % nPoiTypes
        
    
        closestObsDistanceSqr = Inf
        for stepIndex in range(historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = Inf
            for agentIndex in range(number_agents):
                # Calculate separation distance between poi and agent
                separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                distanceSqr = separation0 * separation0 + separation1 * separation1
                
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
    data["Agent Rewards"] = np.ones(number_agents) * globalReward 

#@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

cpdef assignGlobalRewardSimple(data):
    
    cdef int[:,:] itemHeld=data["Item Held"]
    cdef int number_agents = data['Number of Agents']
    
  
    
    cdef int[:]  recipe = data["Recipe"]
    cdef int recipeSize = data["Recipe Size"]
    cdef int ordered    = data["Ordered"] 
    
    cdef int agentIndex, observerCount, poiType, recipeIndex
    
    
    
    cdef double globalReward = 0.0
 
    
    for agentIndex in range(number_agents):
        for recipeIndex in range(recipeSize):
            if ( itemHeld[agentIndex][recipeIndex] == 0):
                break
            if (recipeIndex == recipeSize-1):
                globalReward+=1.0
        
    
    data["Global Reward"] = globalReward
    data["Agent Rewards"] = np.ones(number_agents) * globalReward 


    
#@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef giveKey(data):    
    cdef double obsRadius=data["Observation Radius"] ** 2
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef int coupling=data["Coupling"]
    cdef int couplingLimit=data["Coupling Limit"]
    
    cdef int[:,:] itemHeld = data["Item Held"]
    cdef double[:, :] agentPositionCol = data["Agent Positions"]
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
    
    
    cdef int[:] viewCount=np.zeros(number_pois,dtype=np.int32)
    cdef int[:] indexes=np.zeros(number_agents,dtype=np.int32)
    cdef double[:] dists=np.zeros(number_agents,dtype=np.float64)
    
    cdef int[:]  recipe = data["Recipe"]
    cdef int recipeSize = data["Recipe Size"]
    cdef int nPoiTypes  = data["Number of POI Types"]
    cdef int ordered    = data["Ordered"]  
   
    cdef int agentIndex,  poiIndex, closestIndex, recipeIndex, poiType
    cdef double distanceSqr,closestDist
    
    #determine closest poi to each agent and if each poi is fully viewed
    for agentIndex in range(number_agents):
        closestIndex=-1
        closestDist =1e9
        
        for poiIndex in range(number_pois):
            distanceSqr= (poiPositionCol[poiIndex, 0]-agentPositionCol[agentIndex,0])**2
            distanceSqr+=(poiPositionCol[poiIndex, 1]-agentPositionCol[agentIndex,1])**2
            
            if (distanceSqr<closestDist and viewCount[poiIndex] < couplingLimit):
                closestIndex=poiIndex
                closestDist=distanceSqr
                dists[agentIndex]=closestDist
                indexes[agentIndex]=closestIndex
                
                
        if (closestDist<obsRadius):
            if ordered:
                for recipeIndex in range(recipeSize):
                    
                    if recipe[recipeIndex]==poiType:
                        viewCount[closestIndex]+=1
                        
                    if itemHeld[agentIndex][recipeIndex] == 0:
                        break
                    
            else:
                viewCount[closestIndex]+=1
            
    #for each agent...        
    for agentIndex in range(number_agents):
        closestDist=dists[agentIndex]
        closestIndex=indexes[agentIndex]
        
        #if poi is seen and sufficiently viewed, check if a key can be grabbed
        if (closestDist<obsRadius and viewCount[closestIndex]>=coupling):
            poiType=closestIndex%nPoiTypes
            
            #loop through recipt
            for recipeIndex in range(recipeSize):
                
                #if order doesnt matter and poi in recipe, grab key from poi
                if not ordered:
                    if recipe[recipeIndex]==poiType:
                        itemHeld[agentIndex][recipeIndex]=1
                
                #if order matters,        
                else:
                    #check to see if previous parts of recipe fulfilled
                    if itemHeld[agentIndex][recipeIndex] == 1 or recipe[recipeIndex]==poiType:
                        itemHeld[agentIndex][recipeIndex] = 1
                    
                    #if not, break the loop ans stop checking
                    else: 
                        break
                        
                        
    
#@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef doAgentSenseRecipe(data):
    """
    Sensor model is based off<aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
    Where a means (other) agent, p means poi, and the rest are the quadrants
    The POI section is extended to show different types of POI
    Also has elements for whether or not it has seen each of the POI types (to keep it Markov)
    """
    cdef double obsRadius=data["Observation Radius"] ** 2
    
    
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef double[:, :] agentPositionCol = data["Agent Positions"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
    cdef double[:, :] orientationCol = data["Agent Orientations"]
    
    
   
    cdef int[:] recipe = data["Recipe"]
    cdef int recipeSize = data["Recipe Size"]
    cdef int nPoiTypes   = data["Number of POI Types"]
    cdef int ordered  = data["Ordered"]
    
    cdef int[:,:] itemHeld = data["Item Held"]
    
    #            agent view + poi view + recipe seen+ items grabbed from recipe 
    cdef int obsSize = 4 + 4*nPoiTypes + recipeSize + recipeSize
    
    cdef double[:,:] observationCol = np.zeros((number_agents, obsSize), dtype = np.float64)
    
    
    
    cdef int agentIndex, otherAgentIndex, poiIndex, obsIndex, recipeIndex, poiType, shift
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
            globalFrameSeparation0 = agentPositionCol[otherAgentIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = agentPositionCol[otherAgentIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame using inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
           
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
        	
            
            # other is east of agent
            if agentFrameSeparation0 > 0:
                # other is north-east of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,0] += 1.0 / distanceSqr
                else: # other is south-east of agent
                    observationCol[agentIndex,3] += 1.0  / distanceSqr
            else:  # other is west of agent
                # other is north-west of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,1] += 1.0  / distanceSqr
                else:  # other is south-west of agent
                    observationCol[agentIndex,2] += 1.0  / distanceSqr



        # calculate observation values due to pois
        for poiIndex in range(number_pois):
            
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = poiPositionCol[poiIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = poiPositionCol[poiIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame unp.sing inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
            
            
            poiType=poiIndex % nPoiTypes
            
            shift=poiType*4
            
            
            # poi is east of agent
            if agentFrameSeparation0> 0:
                # poi is north-east of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,4+shift] += poiValueCol[poiIndex]  / distanceSqr
                else: # poi is south-east of agent
                    observationCol[agentIndex,7+shift] += poiValueCol[poiIndex]  / distanceSqr
            else:  # poi is west of agent
                # poi is north-west of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,5+shift] += poiValueCol[poiIndex]  / distanceSqr
                else:  # poi is south-west of agent
                    observationCol[agentIndex,6+shift] += poiValueCol[poiIndex]  / distanceSqr
        
        for recipeIndex in range(recipeSize):
            shift=4+4*nPoiTypes
            
            #recipe requested
            observationCol[agentIndex,shift+recipeIndex] = <double>recipe[recipeIndex]
            
            #keys obtained
            observationCol[agentIndex,shift+recipeIndex+recipeSize] = <double>itemHeld[agentIndex,recipeIndex]
            
            
    data["Agent Observations"] = observationCol
    
    
def doAgentSenseRecipe2(data):
    giveKey(data)    
    doAgentSenseRecipe(data)
