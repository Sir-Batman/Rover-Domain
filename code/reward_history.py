import csv 
import os
import errno
import pandas as pd
import numpy as np

def saveRewardPandas(data):
    """Saves the output rewards as a pandas dataframe in pickle format"""
    saveFileName  = data["Performance Save File Name"]
    # Create File Directory if it doesn't exist
    if not os.path.exists(os.path.dirname(saveFileName)):
        try:
            os.makedirs(os.path.dirname(saveFileName))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    s = pd.Series(data=data["Reward History"])
    s.to_pickle(saveFileName)

    
def saveRewardHistory(data):
    saveFileName  = data["Performance Save File Name"]
    # Create File Directory if it doesn't exist
    if not os.path.exists(os.path.dirname(saveFileName)):
        try:
            os.makedirs(os.path.dirname(saveFileName))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    with open(saveFileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["*Episode"] + list(range(len(data["Reward History"]))))
        writer.writerow(['Performance'] + data["Reward History"])


def createRewardHistory(data):
    data["Reward History"] = []
     
def updateRewardHistory(data):
    data["Reward History"].append(data["Global Reward"])
        
def printGlobalReward(data):
    if data["World Index"] == 0:
        print(data["Global Reward"])
                