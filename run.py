from simulation_specifics import getSim
from simulation_mods import *


# NOTE: Add the mod functions (variables) to run to modCol here:
modCol = [differenceRewardMod]

i = 0
while True:
    print("Run %i"%(i))
    for mod in modCol:
        sim = getSim()
        mod(sim)
        sim.run()
    i += 1
