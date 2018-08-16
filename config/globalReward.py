import datetime

def run(core):
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    
    print("Starting global test at\n\t%s\n"%(dateTimeString))
    
    core.data["Number of Agents"] = 9
    core.data["Number of POIs"] = 4
    core.data["World Width"] = 30.0
    core.data["World Length"] = 30.0
    core.data["Coupling"] = 3
    core.data["Observation Radius"] = 4.0
    core.data["Minimum Distance"] = 1.0
    core.data["Steps"] = 30
    core.data["Trains per Episode"] = 100
    core.data["Tests per Episode"] = 1
    core.data["Number of Episodes"] = 1000
    
    perfSaveFileName = "log/global/perf %s.csv"%(dateTimeString)
    trajSaveFileName = "log/global/traj %s.csv"%(dateTimeString)
    pickleSaveFileName = "log/global/data %s.pickle"%(dateTimeString)

    # NOTE: make sure functions are added to the list in the right order
    
    # print the current Episode
    core.addTrainEndFunc(lambda data: print(data["Episode Index"], data["Global Reward"]))
    core.addTestEndFunc(lambda data: print(data["Episode Index"], data["Global Reward"]))
    
    # Add Rover Domain Construction Functionality
    from code.world_setup import blueprintAgent, blueprintPoi, initWorld
    core.addTrainBeginFunc(blueprintAgent)
    core.addTrainBeginFunc(blueprintPoi)
    core.addWorldTrainBeginFunc(initWorld)
    core.addTestBeginFunc(blueprintAgent)
    core.addTestBeginFunc(blueprintPoi)
    core.addWorldTestBeginFunc(initWorld)
    
    
    # Add Rover Domain Dynamic Functionality (using Cython to speed up code)
    from code.agent_domain_2 import doAgentSense, doAgentProcess, doAgentMove
    core.addWorldTrainStepFunc(doAgentSense)
    core.addWorldTrainStepFunc(doAgentProcess)
    core.addWorldTrainStepFunc(doAgentMove)
    core.addWorldTestStepFunc(doAgentSense)
    core.addWorldTestStepFunc(doAgentProcess)
    core.addWorldTestStepFunc(doAgentMove)
    
    # Add Agent Position Trajectory History Functionality
    from code.trajectory_history import createTrajectoryHistories, updateTrajectoryHistories, saveTrajectoryHistories
    core.addWorldTrainBeginFunc(createTrajectoryHistories)
    core.addWorldTrainStepFunc(updateTrajectoryHistories)
    core.addTrialEndFunc(saveTrajectoryHistories(trajSaveFileName))
    core.addWorldTestBeginFunc(createTrajectoryHistories)
    core.addWorldTestStepFunc(updateTrajectoryHistories)

    
    
    # Add Agent Reward Functionality
    from code.reward import assignGlobalReward, assignDifferenceReward
    core.addWorldTrainEndFunc(assignGlobalReward)
    core.addWorldTestEndFunc(assignGlobalReward)
    
    # Add Performance Recording Functionality
    from code.reward_history import printGlobalReward, saveRewardHistory, createRewardHistory, updateRewardHistory
    core.addTrialBeginFunc(createRewardHistory)
    core.addTestEndFunc(updateRewardHistory)
    core.addTrialEndFunc(saveRewardHistory(perfSaveFileName))
    
    # # Add DE Functionality (all functionality below are dependent and are displayed together for easy accessibility)
    # from code.differential_evolution import initDe, assignDePolicies, rewardDePolicies, evolveDePolicies, assignBestDePolicies
    # core.addTrialBeginFunc(initDe(input_shape= 8, num_outputs=2, num_units = 16))
    # core.addWorldTrainBeginFunc(assignDePolicies)
    # core.addWorldTrainEndFunc(rewardDePolicies)
    # core.addTrainEndFunc(evolveDePolicies)
    # core.addWorldTestBeginFunc(assignBestDePolicies)
    # 
    # Add CCEA Functionality (all functionality below are dependent and are displayed together for easy accessibility)
    from code.ccea import initCcea, assignCceaPolicies, rewardCceaPolicies, evolveCceaPolicies, assignBestCceaPolicies
    core.addTrialBeginFunc(initCcea(input_shape= 8, num_outputs=2, num_units = 16))
    core.addWorldTrainBeginFunc(assignCceaPolicies)
    core.addWorldTrainEndFunc(rewardCceaPolicies)
    core.addTestEndFunc(evolveCceaPolicies)
    core.addWorldTestBeginFunc(assignBestCceaPolicies)
    
    core.run()

    import pickle
    
    with open(pickleSaveFileName, 'wb') as handle:
        pickle.dump(core.data, handle, protocol = pickle.HIGHEST_PROTOCOL)


"""
TODO
make sim_core code
make config code as just values to change
the problem with turning some codes into classes is that I can not easily change 
    the parameters mid trial
give Poi's a value (changes setup, reward and sense functions)
"""

