from policies.policy import RandomPolicy

def assignRandomPolicy(data):
    number_agents = data['Number of Agents']
    policies = [None] * number_agents
    
    for agentIndex in range(data['Number of Agents']):
        policies[agentIndex] = RandomPolicy(2)
    worldData["Agent Policies"] = policies