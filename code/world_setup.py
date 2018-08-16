from numpy import array, random, cos, sin, pi, vstack

def blueprintAgent(data):
    number_agents = data['Number of Agents']
    world_width = data['World Width']
    world_length = data['World Length']
    
    
    # Initial all agents in the randomly in world
    data['Agent Positions BluePrint'] = random.rand(number_agents, 2) * [world_width, world_length]
    angles = random.uniform(-pi, pi, number_agents)
    data['Agent Orientations BluePrint'] = vstack((cos(angles), sin(angles))).T

    
def blueprintPoi(data):
    number_pois = data['Number of POIs']    
    world_width = data['World Width']
    world_length = data['World Length']    
    
    
    # Initialize all Pois Randomly
    data['Poi Positions BluePrint'] = random.rand(number_pois, 2) * [world_width, world_length]
 
def initWorld(data):
    data['Agent Positions'] = data['Agent Positions BluePrint'].copy()
    data['Agent Orientations'] = data['Agent Orientations BluePrint'].copy()
    data['Poi Positions'] = data['Poi Positions BluePrint'].copy()
