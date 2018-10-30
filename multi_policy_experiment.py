import datetime
from simulation_core import SimulationCore
import pyximport; pyximport.install() # For cython(pyx) code
from code.world_setup import * # Rover Domain Construction 
from code.agent_domain_2 import * # Rover Domain Dynamic  
from code.trajectory_history import * # Agent Position Trajectory History 
from code.reward import * # Agent Reward 
from code.reward_history import * # Performance Recording 
from code.ccea import * # CCEA 
from code.save_to_pickle import * # Save data as pickle file
    
def getSim():
    sim = SimulationCore()
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    
    sim.data["Specifics Name"] = "Schedule Tests"#"Schedule_Experiments_9A4P3C50W"

    sim.data["Number of Agents"] = 9
    sim.data["Number of POIs"] = 4
    sim.data["Coupling"] = 3
    sim.data["World Width"] = 50.0
    sim.data["World Length"] = 50.0
    sim.data["Steps"] = 60
    sim.data["Observation Radius"] = 4.0
    sim.data["Minimum Distance"] = 1.0

    sim.data["Trains per Episode"] = 1
    sim.data["Tests per Episode"] = 1
    sim.data["Number of Episodes"] = 1

    # Fixed POI placement positions NOTE MUST EQUAL NUM POI's
    sim.data['Fixed Poi Positions'] = [(5.,5.), (45.,5.), (5,45.), (45.,45.)]

    # Multireward parameters
    sim.data["Policy Schedule"] = [("Team2", 0), ("Team3", 10), ("Team4", 25), ("GoToPOI", 35)]
    sim.data["Test Name"] = "Schedule-T2-0_T3-10_T4-25_POI35"
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Test Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Test Name"], dateTimeString)
        
    # sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"% (sim.data["Specifics Name"], sim.data["Test Name"], dateTimeString)
        

    # NOTE: make sure FuncCol.appendtions are added to the list in the right order
    
    # print the current Episode
    sim.testEndFuncCol.append(lambda data: print(data["Episode Index"], data["Global Reward"]))
    sim.trialEndFuncCol.append(lambda data: print())
    
    
    # Add Rover Domain Construction Functionality
    sim.trainBeginFuncCol.append(blueprintAgent)
    sim.trainBeginFuncCol.append(blueprintPoi)
    sim.worldTrainBeginFuncCol.append(initWorld)
    sim.worldTrainBeginFuncCol.append(staticPOIPlacement) # Hacky Override the initWorld POI placement
    sim.testBeginFuncCol.append(blueprintAgent)
    sim.testBeginFuncCol.append(blueprintPoi)
    sim.worldTestBeginFuncCol.append(initWorld)
    sim.worldTestBeginFuncCol.append(staticPOIPlacement) # Hacky Override the initWorld POI placement
    
    # Add Rover Domain Dynamic Functionality (using Cython to speed up code)
    # Note: Change the Process functions to change the agent type.
    sim.worldTrainStepFuncCol.append(doAgentSense)
    sim.worldTrainStepFuncCol.append(doAgentProcess_ScheduleMP)
    sim.worldTrainStepFuncCol.append(doAgentMove)

    sim.worldTestStepFuncCol.append(doAgentSense)
    sim.worldTestStepFuncCol.append(doAgentProcess_ScheduleMP)
    sim.worldTestStepFuncCol.append(doAgentMove)
    
    # Add Agent Position Trajectory History Functionality
    sim.worldTrainBeginFuncCol.append(createTrajectoryHistories)
    sim.worldTrainStepFuncCol.append(updateTrajectoryHistories)
    sim.trialEndFuncCol.append(saveTrajectoryHistories)
    sim.worldTestBeginFuncCol.append(createTrajectoryHistories)
    sim.worldTestStepFuncCol.append(updateTrajectoryHistories)
    
    # Add Agent Reward Functionality
    sim.worldTrainEndFuncCol.append(assignGlobalReward)
    sim.worldTestEndFuncCol.append(assignGlobalReward)
    
    # Add Performance Recording Functionality
    sim.trialBeginFuncCol.append(createRewardHistory)
    sim.testEndFuncCol.append(updateRewardHistory)
    sim.trialEndFuncCol.append(saveRewardHistory)
    
    # # Add DE Functionality (all Functionality below are dependent and are displayed together for easy accessibility)
    # from code.differential_evolution import initDe, assignDePolicies, rewardDePolicies, evolveDePolicies, assignBestDePolicies
    # sim.trialBeginFuncCol.append(initDe(input_shape= 8, num_outputs=2, num_units = 16))
    # sim.worldTrainBeginFuncCol.append(assignDePolicies)
    # sim.worldTrainEndFuncCol.append(rewardDePolicies)
    # sim.trainEndFuncCol.append(evolveDePolicies)
    # sim.worldTestBeginFuncCol.append(assignBestDePolicies)
    
    # Add CCEA Functionality 
    
    # sim.trialBeginFuncCol.append(initCcea(input_shape= 8, num_outputs=2, num_units = 16))
    # sim.worldTrainBeginFuncCol.append(assignCceaPolicies)
    # sim.worldTrainEndFuncCol.append(rewardCceaPolicies)
    # sim.worldTestBeginFuncCol.append(assignBestCceaPolicies)
    # sim.testEndFuncCol.append(evolveCceaPolicies)

    # Add multi-policy agent structure to world
    sim.trialBeginFuncCol.append(blueprintMultipolicyAgent)
    
    
    # Save data as pickle file
    # sim.trialEndFuncCol.append(savePickle)
    
    return sim

    


if __name__ == "__main__":
    sim= getSim()
    sim.run()
