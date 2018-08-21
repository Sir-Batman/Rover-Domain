def setCurriculumCoupling(data):
    schedule = data["Schedule"]
    episodeIndex = data["Episode Index"]
    generationSum = 0
    trainingCoupling = 1
    for coupling, duration in schedule:
        generationSum += duration
        trainingCoupling = coupling
        if generationSum > episodeIndex:
            break
    data["Test Coupling"] = data["Coupling"] 
    data["Coupling"] = trainingCoupling
            
def restoreCoupling(data):
    data["Coupling"] = data["Test Coupling"]