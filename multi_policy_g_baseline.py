import datetime
from simulation_core import SimulationCore
import pyximport; pyximport.install() # For cython(pyx) code
from src.world_setup import * # Rover Domain Construction
from src.agent_domain_2 import * # Rover Domain Dynamic
from src.trajectory_history import * # Agent Position Trajectory History
from src.reward import * # Agent Reward
from src.reward_history import * # Performance Recording
from src.ccea import * # CCEA
from src.save_to_pickle import * # Save data as pickle file
import uuid

def getSim():
    sim = SimulationCore()

    sim.data["Test Name"] = "Train G Baseline"
    sim.data["Number of Agents"] = 12
    sim.data["Number of POIs"] = 4
    sim.data["World Width"] = 50.0
    sim.data["World Length"] = 50.0
    sim.data["Coupling"] = 3
    sim.data["Observation Radius"] = 4.0
    sim.data["Minimum Distance"] = 1.0
    sim.data["Steps"] = 60
    sim.data["Trains per Episode"] = 2
    sim.data["Tests per Episode"] = 0
    sim.data["Number of Episodes"] = 5000
    sim.data["Specifics Name"] = "G"
    sim.data["Goal Team Size"] = 4
    sim.data["Coupling"] = 4

    # Fixed POI placement positions NOTE MUST EQUAL NUM POI's
    sim.data['Fixed Poi Positions'] = [(5., 5.), (5., 15.), (5, 25.), (5., 45.)]
    unique_id = str(uuid.uuid4())

    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)
        

    # NOTE: make sure FuncCol.appendtions are added to the list in the right order
    
    # print the current Episode
    # sim.testEndFuncCol.append(lambda data: print(data["Episode Index"], data["Global Reward"]))
    # sim.trialEndFuncCol.append(lambda data: print())
    
    
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
    sim.worldTrainStepFuncCol.append(doAgentProcess)
    sim.worldTrainStepFuncCol.append(doAgentMove)

    sim.worldTestStepFuncCol.append(doAgentSense)
    sim.worldTestStepFuncCol.append(doAgentProcess)
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
    sim.trialEndFuncCol.append(saveRewardPandas)
    
    # # Add DE Functionality (all Functionality below are dependent and are displayed together for easy accessibility)
    # from src.differential_evolution import initDe, assignDePolicies, rewardDePolicies, evolveDePolicies, assignBestDePolicies
    # sim.trialBeginFuncCol.append(initDe(input_shape= 8, num_outputs=2, num_units = 16))
    # sim.worldTrainBeginFuncCol.append(assignDePolicies)
    # sim.worldTrainEndFuncCol.append(rewardDePolicies)
    # sim.trainEndFuncCol.append(evolveDePolicies)
    # sim.worldTestBeginFuncCol.append(assignBestDePolicies)
    
    # Add CCEA Functionality 
    
    sim.trialBeginFuncCol.append(initCcea(input_shape= 8, num_outputs=2, num_units = 16))
    sim.worldTrainBeginFuncCol.append(assignCceaPolicies)
    sim.worldTrainEndFuncCol.append(rewardCceaPolicies)
    sim.worldTestBeginFuncCol.append(assignBestCceaPolicies)
    sim.testEndFuncCol.append(evolveCceaPolicies)

    # Add multi-policy agent structure to world
    # sim.trialBeginFuncCol.append(blueprintMultipolicyAgent)
    
    
    # Save data as pickle file
    sim.trialEndFuncCol.append(savePickle)
    
    return sim


def trial(i):
    print("Starting trial {}".format(i))
    sim = getSim()
    sim.run()


if __name__ == "__main__":
    import multiprocessing
    with multiprocessing.Pool(8) as p:
        p.map(trial, range(100))


