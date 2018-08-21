import datetime
from code.reward import * # Agent Reward 
from code.curriculum import * # Agent Curriculum


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
        
        
def globalRewardC1Mod(sim):
    sim.data["Mod Name"] = "global_coupling_curriculum"
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting global coupling curriculum test at\n\t%s\n"%(dateTimeString))
    
    # Agent Reward 
    sim.worldTrainEndFuncCol = [assignGlobalReward if func == assignGlobalReward \
        else func for func in sim.worldTrainEndFuncCol] 
    
    # Agent Curriculum
    sim.data["Schedule"] = ((1,15),(2,25),(3,50),(4,200),(5,1000))
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardC1Mod(sim):
    sim.data["Mod Name"] = "difference_coupling_curriculum"
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting difference coupling curriculum test at\n\t%s\n"%(dateTimeString))
    
    # Agent Reward 
    sim.worldTrainEndFuncCol = [assignDifferenceReward if func == assignGlobalReward \
        else func for func in sim.worldTrainEndFuncCol] 
    
    # Agent Curriculum
    sim.data["Schedule"] = ((1,15),(2,25),(3,50),(4,200),(5,1000))
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        