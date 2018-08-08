import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from torch import Tensor

class Policy(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def get_next(self, state, *args, **kwargs):
        pass
    
    def get_train(self, *args, **kwargs):
        pass
    

class RandomPolicy(Policy):
    def __init__(self, output_shape, low=-1, high=1):
        self.output_shape = output_shape
        self.low = low
        self.high = high

    def get_next(self,state):
        return np.random.uniform(self.low, self.high, self.output_shape)
        
class Evo_MLP(nn.Module, Policy):
    def __init__(self, input_shape, num_outputs, num_units=16):
        super(Evo_MLP, self).__init__()
        
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.num_units = num_units
        self.fitness = float("-inf")

        self.fc1 = nn.Linear(input_shape, num_units)
        self.fc2 = nn.Linear(num_units, num_outputs)
        for param in self.parameters():
            param.requires_grad = False

    def get_next(self, state):
        x = Variable(torch.FloatTensor(state))
        x = F.relu(self.fc1(x))
        y = F.tanh(self.fc2(x))
        return np.array(y)

    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, 2)

    def mutate(self):
        random_w1 = np.random.normal(0, 10, list(self.fc1.weight.size()))
        random_w1 *= (np.random.uniform(size = list(self.fc1.weight.size())) < 0.1).astype(float)
        random_w2 = np.random.normal(0, 10, list(self.fc2.weight.size()))
        random_w2 *= (np.random.uniform(size = list(self.fc2.weight.size())) < 0.1).astype(float)
        self.fc1.weight += Tensor(random_w1)
        self.fc2.weight += Tensor(random_w2)
        
    def copy(self):
        newMlp = Evo_MLP(self.input_shape, self.num_outputs, self.num_units)
        newMlp.load_state_dict(self.state_dict())
        return newMlp

def CCEA(population, fitness, retain=0.2):
    '''
    Implements the basic CCEA algorithm with binary tournaments.
    Maintains 20% of the population each time by default.
    '''
    population_size = population.num_agents
    #print(population, fitness)
    scored_pop = []
    for agent_id in population.agent_policies:
        scored_pop.append((population.agent_policies[agent_id], fitness[agent_id]))
    scored_pop = sorted(scored_pop, key=lambda x: x[1],reverse=True)
    scored_pop = [x[0] for x in scored_pop]
    # retain x% of the population
    scored_pop = scored_pop[:int(population_size*retain)]
    for _ in range(population_size - len(scored_pop)):
        # We only choose from the unmutated survivors
        choice = random.choice(scored_pop)
        #print(choice)
        choice.mutate()
        scored_pop.append(choice)
    # Return the evaluated population.
    return scored_pop
    
def initCcea(input_shape, num_outputs, num_units=16):
    def initCceaGo(data):
        number_agents = data['Number of Agents']
        populationCol = [[Evo_MLP(input_shape,num_outputs,num_units) for i in range(data['Trains per Episode'])] for j in range(number_agents)] 
        data['Agent Populations'] = populationCol
    return initCceaGo
    
def assignCceaPolicies(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    worldIndex = data["World Index"]
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = populationCol[agentIndex][worldIndex]
    data["Agent Policies"] = policyCol
    
def assignBestCceaPolicies(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = max(populationCol[agentIndex], key = lambda policy: policy.fitness)
    data["Agent Policies"] = policyCol

def rewardCceaPolicies(data):
    policyCol = data["Agent Policies"]
    number_agents = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agentIndex in range(number_agents):
        policyCol[agentIndex].fitness = rewardCol[agentIndex]
    
def evolveCceaPolicies(data): 
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    for agentIndex in range(number_agents):
        population = populationCol[agentIndex]
        newPopulation = [None] * len(population)
        
        random.shuffle(population)
        
        # Binary Tournament
        newPolicyIndex = 0
        for matchIndex in range(len(population)//2):
            if population[2 * matchIndex].fitness > population[2 * matchIndex + 1].fitness:
                newPopulation[matchIndex] = population[2 * matchIndex]
            else:
                newPopulation[matchIndex] = population[2 * matchIndex + 1]
            newPolicyIndex += 1
            
        # Random fill with mutation
        elite = newPopulation[:newPolicyIndex]
        while newPolicyIndex < len(population):
            newPolicy = random.choice(elite).copy()
            newPolicy.mutate()
            newPopulation[newPolicyIndex] = newPolicy
            newPolicyIndex += 1
            
        data['Agent Populations'][agentIndex] = newPopulation
        