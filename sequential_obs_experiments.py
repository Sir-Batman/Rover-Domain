from simulation_core import SimulationCore
from uuid import uuid4

import pyximport
pyximport.install()  # For cython(pyx) code

from src.world_setup import *  # Rover Domain Construction
from src.agent_domain_2 import *  # Rover Domain Dynamic
from src.trajectory_history import *  # Agent Position Trajectory History
from src.reward import *  # Agent Reward
from src.reward_history import *  # Performance Recording
from src.ccea import *  # CCEA
from src.save_to_pickle import *  # Save data as pickle file
import numpy as np
import pandas as pd
import poi_recipes


def makeSim():
    """
    Makes the simulator object for this series of experiments.
    :return: sim: SimulationCore object with general parameters set.
    """
    sim = SimulationCore()

    # Experiment Parameters
    sim.data["Test Name"] = "Sequential Obs"
    # World Parameters
    sim.data["World Width"] = 50.0
    sim.data["World Length"] = 50.0
    sim.data["Steps"] = 60
    # Agent Parameters
    sim.data["Number of Agents"] = 12
    # POI Parameters
    sim.data["Number of POIs"] = 4
    sim.data["Goal Team Size"] = 4
    sim.data["Coupling"] = 3
    sim.data["Observation Radius"] = 4.0
    sim.data["Minimum Distance"] = 1.0
    sim.data['Fixed Poi Positions'] = [(5., 5.), (5., 15.), (5, 25.), (5., 45.)]
    # Learning Parameters
    sim.data["Trains per Episode"] = 2
    sim.data["Tests per Episode"] = 0
    sim.data["Number of Episodes"] = 5000

    # Training functions
    sim.trainBeginFuncCol.append(blueprintAgent)
    sim.trainBeginFuncCol.append(blueprintPoi)

    sim.worldTrainBeginFuncCol.append(initWorld)
    sim.worldTrainBeginFuncCol.append(staticPOIPlacement)  # Hacky Override the initWorld POI placement
    sim.worldTrainBeginFuncCol.append(createTrajectoryHistories)

    sim.worldTrainStepFuncCol.append(poi_recipes.doAgentSenseMod)
    sim.worldTrainStepFuncCol.append(doAgentProcess)
    sim.worldTrainStepFuncCol.append(doAgentMove)
    sim.worldTrainStepFuncCol.append(updateTrajectoryHistories)

    # sim.worldTrainEndFuncCol.append(assignDifferenceReward)

    # Testing functions
    sim.testBeginFuncCol.append(blueprintAgent)
    sim.testBeginFuncCol.append(blueprintPoi)

    sim.worldTestBeginFuncCol.append(initWorld)
    sim.worldTestBeginFuncCol.append(staticPOIPlacement)  # Hacky Override the initWorld POI placement
    sim.worldTestBeginFuncCol.append(createTrajectoryHistories)

    sim.worldTestStepFuncCol.append(doAgentSense)
    sim.worldTestStepFuncCol.append(doAgentProcess)
    sim.worldTestStepFuncCol.append(doAgentMove)
    sim.worldTestStepFuncCol.append(updateTrajectoryHistories)

    # sim.worldTestEndFuncCol.append(assignDifferenceReward)

    # Etc. Functions
    # Save data as pickle file
    sim.trialEndFuncCol.append(savePickle)

    return sim


def g_trial(sim):
    """G baseline comparison"""
    print("Starting trial ")
    sim.data["Specifics Name"] = "G"
    unique_id = str(uuid4())
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf_%s.pkl" % \
                                             (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)

    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv" % \
                                            (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)

    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle" % \
                                        (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)
    sim.worldTrainEndFuncCol.append(assignGlobalReward)
    sim.worldTestEndFuncCol.append(assignGlobalReward)

    # Add CCEA Functionality (Split across Trial/Train sets)
    sim.trialBeginFuncCol.append(initCcea(input_shape=8, num_outputs=2, num_units=16))

    sim.worldTrainBeginFuncCol.append(assignCceaPolicies)

    sim.worldTrainEndFuncCol.append(rewardCceaPolicies)

    sim.worldTestBeginFuncCol.append(assignBestCceaPolicies)

    sim.testEndFuncCol.append(evolveCceaPolicies)

    return sim


def d_trial(sim):
    """D baseline comparison"""
    print("Starting trial ")
    sim.data["Specifics Name"] = "D"
    unique_id = str(uuid4())
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf_%s.pkl" % \
                                             (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)

    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv" % \
                                            (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)

    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle" % \
                                        (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)
    sim.worldTrainEndFuncCol.append(assignDifferenceReward)
    sim.worldTestEndFuncCol.append(assignDifferenceReward)

    # Add CCEA Functionality (Split across Trial/Train sets)
    sim.trialBeginFuncCol.append(initCcea(input_shape=8, num_outputs=2, num_units=16))

    sim.worldTrainBeginFuncCol.append(assignCceaPolicies)

    sim.worldTrainEndFuncCol.append(rewardCceaPolicies)

    sim.worldTestBeginFuncCol.append(assignBestCceaPolicies)

    sim.testEndFuncCol.append(evolveCceaPolicies)

    return sim


def alignment_trial(sim):
    """Main alignment test"""
    # TODO Make this based on the alignment between the rewards, alignment by default
    sim.data["Specifics Name"] = "Alignment"
    unique_id = str(uuid4())
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf_%s.pkl" % \
                                             (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)

    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv" % \
                                            (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)

    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle" % \
                                        (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)
    return sim


def perfect_policy(sim):
    """Hand-coded optimal policy"""
    # TODO code this based on the state relation (similar to the previous one) and just make it select the 3
    # TODO policies by if the agent has observed things in the perfect order
    # Used to calculate the empirical top score
    sim.data["Specifics Name"] = "Perfect"
    unique_id = str(uuid4())
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf_%s.pkl" % \
                                             (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)

    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv" % \
                                            (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)

    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle" % \
                                        (sim.data["Specifics Name"], sim.data["Test Name"], unique_id)
    return sim


def trial(i):
    """Applies modification and then runs. Worker function for multiprocessing."""
    print("Starting trial {}".format(i['info']))
    sim = makeSim()
    sim = i['mod'](sim)
    sim.run()


if __name__ == "__main__":
    import multiprocessing
    with multiprocessing.Pool(5) as p:
        p.map(trial, range(100))
