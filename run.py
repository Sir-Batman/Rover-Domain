from specifics import getSim
from mods import *
from shutil import copyfile

# NOTE: Add the mod functions (variables) to run to modCol here:
modCol = [
    globalRewardMod,
    differenceRewardMod,
    globalRewardC1Mod,
    differenceRewardC1Mod
]

def copyTestFiles():
    copyfile("mods.py", "log/%s/mods.py"%sim.data["Specifics Name"])
    copyfile("specifics.py", "log/%s/specifics.py"%sim.data["Specifics Name"])

i = 0
while True:
    print("Run %i"%(i))
    for mod in modCol:
        sim = getSim()
        mod(sim)
        sim.run()
    i += 1


# def main():
#     sim = getSim()
#     globalRewardMod(sim)
#     sim.run()