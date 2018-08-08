from numpy import array, random, cos, sin, pi

def blueprintAgent(data):
    number_agents = data['Number of Agents']
    world_width = data['World Width']
    world_length = data['World Length']
    
    
    # Initial all agents in the center of the world
    data['Agent Positions BluePrint'] = [None] * number_agents
    data['Agent Orientations BluePrint'] = [None] * number_agents
    for agentIndex in range(number_agents):
        data['Agent Positions BluePrint'][agentIndex] = array([world_width/2.0, world_length/2.0])
        angle = random.uniform(-pi, pi)
        data['Agent Orientations BluePrint'][agentIndex] = array([cos(angle), sin(angle)])

def blueprintPoi(data):
    number_pois = data['Number of POIs']    
    world_width = data['World Width']
    world_length = data['World Length']    
    
    
    # Initialize all Pois Randomly
    data['Poi Positions BluePrint'] = [None] * number_pois
    for poiIndex in range(number_pois):
        data['Poi Positions BluePrint'][poiIndex] = random.rand(2) * [world_width, world_length]
        
def initWorld(data):
    data['Agent Positions'] = data['Agent Positions BluePrint']
    data['Agent Orientations'] = data['Agent Orientations BluePrint']
    data['Poi Positions'] = data['Poi Positions BluePrint'] 