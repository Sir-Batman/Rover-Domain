from simulation_core import SimulationCore

import config.globalReward as g
import config.differenceReward as d

# TODO, load a single config file for quick editting


i = 0
while True:
    print("Test%d"%(i))
    core = SimulationCore()
    g.run(core)
    
    print("Test%d"%(i))
    core = SimulationCore()
    d.run(core)
    
    i += 1
    