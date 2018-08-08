import csv
import os

def createTrajectoryHistories(data):
    # create a history of positions for each agents in order to evalutate reward at the end
    number_agents = data['Number of Agents']
    historyStepCount = data["Steps"] + 1
    agentPositionHistory = [[None for i in range(historyStepCount)] for j in range(number_agents)]
    agentOrientationHistory = [[None for i in range(historyStepCount)] for j in range(number_agents)]
    positionCol = data["Agent Positions"]
    orientationCol = data["Agent Orientations"]
    
    # populate history with current positions
    for agentIndex in range(number_agents):
        agentPositionHistory[agentIndex][0] = positionCol[agentIndex]
        agentOrientationHistory[agentIndex][0] = orientationCol[agentIndex]
    
    
    data["Agent Position History"] = agentPositionHistory
    data["Agent Orientation History"] = agentOrientationHistory
    
    
def updateTrajectoryHistories(data):
    number_agents = data['Number of Agents']
    stepIndex = data["Step Index"]
    historyStepCount = data["Steps"] + 1
    agentPositionHistory = data["Agent Position History"]
    agentOrientationHistory = data["Agent Orientation History"]
    positionCol = data["Agent Positions"]
    orientationCol = data["Agent Orientations"]
    
    # populate history with current positions
    for agentIndex in range(number_agents):
        agentPositionHistory[agentIndex][stepIndex + 1] = positionCol[agentIndex]
        agentOrientationHistory[agentIndex][stepIndex + 1] = orientationCol[agentIndex]
        
    data["Agent Position History"] = agentPositionHistory
    data["Agent Orientation History"] = agentOrientationHistory
    
def saveTrajectoryHistories(saveFileName):
    def saveTrajectoryHistoriesGo(data):
        number_agents = data['Number of Agents']
        number_pois = data["Number of POIs"]
        historyStepCount = data["Steps"] + 1
        agentPositionHistory = data["Agent Position History"]
        agentOrientationHistory = data["Agent Orientation History"]
        poiPositionCol = data["Poi Positions"]
        
        if not os.path.exists(os.path.dirname(saveFileName)):
            try:
                os.makedirs(os.path.dirname(saveFileName))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        with open(saveFileName, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            for agentIndex in range(number_agents):
                writer.writerow(["Agent %d Position 0"%(agentIndex)] + [pos[0] for pos in agentPositionHistory[agentIndex]])
                writer.writerow(["Agent %d Position 1"%(agentIndex)] + [pos[1] for pos in agentPositionHistory[agentIndex]])
                writer.writerow(["Agent %d Orientation 0"%(agentIndex)] + [ori[0] for ori in agentOrientationHistory[agentIndex]])
                writer.writerow(["Agent %d Orientation 1"%(agentIndex)] + [ori[1] for ori in agentOrientationHistory[agentIndex]])
                
            for poiIndex in range(number_pois):
                writer.writerow(["Poi %d Position 0"%(agentIndex)] + [poiPositionCol[poiIndex][0]])
                writer.writerow(["Poi %d Position 1"%(agentIndex)] + [poiPositionCol[poiIndex][1]])

    return saveTrajectoryHistoriesGo