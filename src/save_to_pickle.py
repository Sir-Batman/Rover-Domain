import pickle
import os

def savePickle(data):
    saveFileName = data["Pickle Save File Name"]
    
    if not os.path.exists(os.path.dirname(saveFileName)):
        try:
            os.makedirs(os.path.dirname(saveFileName))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    with open(saveFileName, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)