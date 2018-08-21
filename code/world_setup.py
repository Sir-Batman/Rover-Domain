import numpy as np

def blueprintAgent(data):
    number_agents = data['Number of Agents']
    world_width = data['World Width']
    world_length = data['World Length']
    
    # Initial all agents in the np.randomly in world
    data['Agent Positions BluePrint'] = np.random.rand(number_agents, 2) * [world_width, world_length]
    angles = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angles), np.sin(angles))).T

    
def blueprintPoi(data):
    number_pois = data['Number of POIs']    
    world_width = data['World Width']
    world_length = data['World Length']  
    
    # Initialize all Pois np.randomly
    data['Poi Positions BluePrint'] = np.random.rand(number_pois, 2) * [world_width, world_length]
    data['Poi Values BluePrint'] = np.arange(number_pois) + 1.0
 
 
def initWorld(data):
    data['Agent Positions'] = data['Agent Positions BluePrint'].copy()
    data['Agent Orientations'] = data['Agent Orientations BluePrint'].copy()
    data['Poi Positions'] = data['Poi Positions BluePrint'].copy()
    data['Poi Values'] = data['Poi Values BluePrint'].copy()