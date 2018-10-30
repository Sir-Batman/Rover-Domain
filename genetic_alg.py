from deap import base, creator, tools, algorithms
import numpy as np
import random
import multi_policy_experiment as mpe

class geneticPolicySearch(object):
    def __init__(self, timesteps, getSim):
        # Creator variables
        self.getSim = getSim
        self.timesteps = timesteps
        self.t = 0
        creator.create("FitnessMax", base.Fitness, weights=(0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_policy", self._create_value)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_policy, n=timesteps)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register toolbox functions
        self.toolbox.register("evaluate", self._get_fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate, indpb=1/3.0)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_value(self):
        if self.t == self.timesteps:
            self.t = 0
        p = random.choice(["GoToPOI", "Team2", "Team3", "Team4"])
        v = (p, self.t)
        self.t += 1
        return v

    def _mutate(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() <= indpb:
                p = random.choice(["GoToPOI", "Team2", "Team3", "Team4"])
                v = (p, individual[i][1]) # Copy the timestep
                individual[i] = v
        return individual, 

    def _get_fitness(self, individual):
        ''' will call the simulator on this world. '''
        sim = self.getSim()
        sim.data["Policy Schedule"] = individual
        sim.data["Test Name"] = "Genetic Search "+str(individual)
        sim.run()
        return [sim.data["Global Reward"]]

# Run main search
GENS = 10
search = geneticPolicySearch(50, mpe.getSim)
pop = search.toolbox.population(n=10)
hof = tools.HallOfFame(maxsize=1)
algorithms.eaSimple(population=pop, toolbox=search.toolbox, cxpb=0.5, mutpb=0.2, ngen=50, halloffame=hof, verbose=True)
print("SOLUTION:", hof[0])
