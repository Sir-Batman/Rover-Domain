import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from torch import Tensor



class RandomPolicy:
    def __init__(self, output_shape, low=-1, high=1):
        self.output_shape = output_shape
        self.low = low
        self.high = high

    def get_next(self,state):
        return np.random.uniform(self.low, self.high, self.output_shape)
        
class Evo_MLP(nn.Module):
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
        x = self.fc1(x).tanh()
        y = self.fc2(x).tanh()
        return y.numpy()
        
    def init_weights(m):
        # Not implemented
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, 2)

    def mutate(self):
        m = 10
        mr = 0.01
        random_w1 = np.random.normal(0, m, list(self.fc1.weight.size()))
        random_w1 *= (np.random.uniform(size = list(self.fc1.weight.size())) < mr).astype(float)
        random_w2 = np.random.normal(0, m, list(self.fc2.weight.size()))
        random_w2 *= (np.random.uniform(size = list(self.fc2.weight.size())) < mr).astype(float)
        self.fc1.weight += Tensor(random_w1)
        self.fc2.weight += Tensor(random_w2)
        
    def copy(self):
        newMlp = Evo_MLP(self.input_shape, self.num_outputs, self.num_units)
        newMlp.load_state_dict(self.state_dict())
        return newMlp
    
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
        #policyCol[agentIndex] = populationCol[agentIndex][0]
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
            
        random.shuffle(newPopulation)
        data['Agent Populations'][agentIndex] = newPopulation
        