
class SimulationCore:
    def __init__(self):
        self.data = {
            "Steps": 5,
            "Trains per Episode": 3,
            "Tests per Episode": 1,
            "Number of Episodes": 20
        } 
    
        self.trialBeginFuncCol = []
        
        self.trainBeginFuncCol = []
        self.worldTrainBeginFuncCol = []
        self.worldTrainStepFuncCol = []
        self.worldTrainEndFuncCol = []
        self.trainEndFuncCol = []
        
        self.testBeginFuncCol = []
        self.worldTestBeginFuncCol = []
        self.worldTestStepFuncCol = []
        self.worldTestEndFuncCol = []
        self.testEndFuncCol = []
        
        self.trialEndFuncCol = []
        
    def run(self):
        # Do Trial Begin Functions
        for func in self.trialBeginFuncCol:
            func(self.data)
            
        # Do Each Episode
        for episodeIndex in range(self.data["Number of Episodes"]):
            self.data["Episode Index"] = episodeIndex
            
            # Do Begin Training Functions
            for func in self.trainBeginFuncCol:
                func(self.data)
    
            # Repeat running world (with new teams) until repeat is set to false
            for worldIndex in range(self.data["Trains per Episode"]):
                self.data["Mode"] = "Train"
                self.data["World Index"] = worldIndex
                self.data["Step Index"] = None
                
                # Do world begin (setup) functions
                for func in self.worldTrainBeginFuncCol:
                    func(self.data)
                
                # Do world end functions
                for stepIndex in range(self.data["Steps"]):
                    self.data["Step Index"] = stepIndex
                    for func in self.worldTrainStepFuncCol:
                        func(self.data)
                    
                # Do world 
                for func in self.worldTrainEndFuncCol:
                    func(self.data)
    
            # Do End Training Functions
            for func in self.trainEndFuncCol:
                func(self.data)
            
            # Do Begin Testing Functions
            for func in self.testBeginFuncCol:
                func(self.data)
    
            # Repeat running world (with new teams) until repeat is set to false
            for worldIndex in range(self.data["Tests per Episode"]):
                self.data["Mode"] = "Test"
                self.data["World Index"] = worldIndex
                self.data["Step Index"] = None
                
                
                # Do world begin (setup) functions
                for func in self.worldTestBeginFuncCol:
                    func(self.data)
                
                # Do world end functions
                for stepIndex in range(self.data["Steps"]):
                    self.data["Step Index"] = stepIndex
                    for func in self.worldTestStepFuncCol:
                        func(self.data)
                    
                # Do world 
                for func in self.worldTestEndFuncCol:
                    func(self.data)
    
            # Do End Testing Functions
            for func in self.testEndFuncCol:
                func(self.data)
                
        # Do Trial End Functions
        for func in self.trialEndFuncCol:
            func(self.data)
            
    def addTrialBeginFunc(self, func):
        self.trialBeginFuncCol.append(func)


    def addTrainBeginFunc(self, func):
        self.trainBeginFuncCol.append(func)

    def addWorldTrainBeginFunc(self, func):
        self.worldTrainBeginFuncCol.append(func)
        
    def addWorldTrainStepFunc(self, func):
        self.worldTrainStepFuncCol.append(func)  
        
    def addWorldTrainEndFunc(self, func):
        self.worldTrainEndFuncCol.append(func)  
    
    def addTrainEndFunc(self, func):
        self.trainEndFuncCol.append(func) 
        
    def addTestBeginFunc(self, func):
        self.testBeginFuncCol.append(func) 
        

    def addWorldTestBeginFunc(self, func):
        self.worldTestBeginFuncCol.append(func)
        
        
    def addWorldTestStepFunc(self, func):
        self.worldTestStepFuncCol.append(func) 
        
    def addWorldTestEndFunc(self, func):
        self.worldTestEndFuncCol.append(func) 
        
    def addTestEndFunc(self, func):
        self.testEndFuncCol.append(func) 
        
    def addTrialEndFunc(self, func):
        self.trialEndFuncCol.append(func) 
        


