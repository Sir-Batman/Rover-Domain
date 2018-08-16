from simulation_core import SimulationCore
import numpy
import pyximport
pyximport.install(setup_args={'include_dirs': numpy.get_include()})
import config.globalReward as g
import config.differenceReward as d

 
i = 0
while True:
    print("Test%d"%(i))
    core = SimulationCore()
    g.run(core)
    
    print("Test%d"%(i))
    core = SimulationCore()
    d.run(core)
    
    i += 1
