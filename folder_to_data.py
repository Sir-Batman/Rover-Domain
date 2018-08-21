import csv
import glob
from numpy import *
from scipy import stats
import matplotlib.pyplot as plt

data = {}

def folderToData(folderName):
    
    rawData = {}

    # Get all data files in folder
    fileNameCol = glob.glob(folderName + "*.csv")
    sampleCount = len(fileNameCol)
    averagingLabelCol = []
    allLabelCol = []
    
    # Get all labels from first file
    with open(fileNameCol[0], newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            label = row[0]
            
            # stared labels are labels we do not average
            if label[0] == '*':
                label = label[1:]
                rawData[label] = array(list(map(float, row[1:])))
            else:
                rawData[label] = zeros((sampleCount, len(row[1:])))
                averagingLabelCol.append(label)
                allLabelCol.append(label + "_err")
            allLabelCol.append(label)
                
    # Get all data from files
    sampleIndex = 0
    for fileName in fileNameCol:
        with open(fileName, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                label = row[0]
                if label[0] != '*':
                    rawData[label][sampleIndex, :]  = list(map(float, row[1:]))
        sampleIndex += 1
        
    # Get error and mean (deleting raw data in the process)
    for label in averagingLabelCol:
        rawData[label + "_err"] = stats.sem(rawData[label])
        rawData[label] = mean(rawData[label], axis = 0)
    
    # Use spacing to lower resolution of data
    for label in allLabelCol:
        rawData[label] = rawData[label]
        
    return rawData
    
