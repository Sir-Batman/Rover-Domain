from numpy import array, random, cos, sin, pi, vstack
import code.subpolicies as subp

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

def staticPOIPlacement(data):
    data['Poi Positions'] = data['Fixed Poi Positions']

def blueprintMultipolicyAgent(data):
    """
    Sets up the agent policy strcutre.

    Preconditions: 
    """
    agent_policy_list = []
    for a in range(data['Number of Agents']):
        policies = {}
        # LOAD ALL THE POLICIES, one line for each policy type
        policies["GoToPOI"] = subp.poi_policy
        policies["GoToRover"] = subp.agent_policy
        policies["Team2"] = subp.team_2_policy()
        policies["Team3"] = subp.team_3_policy()
        policies["Team4"] = subp.team_4_policy()
        policies["Random"] = subp.random_policy
        agent_policy_list.append(policies)

    data['Agent Policies'] = agent_policy_list
    
