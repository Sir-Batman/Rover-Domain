import datetime
from code.reward import * # Agent Reward 


def globalRewardMod(sim):
    sim.data["Mod Name"] = "global"
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting global test at\n\t%s\n"%(dateTimeString))
    
    # Agent Reward 
    sim.worldTrainEndFuncCol = [assignGlobalReward if func == assignGlobalReward \
        else func for func in sim.worldTrainEndFuncCol] 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardMod(sim):
    sim.data["Mod Name"] = "difference"
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting difference test at\n\t%s\n"%(dateTimeString))
    
    # Agent Reward 
    sim.worldTrainEndFuncCol = [assignDifferenceReward if func == assignGlobalReward \
        else func for func in sim.worldTrainEndFuncCol] 
    
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        