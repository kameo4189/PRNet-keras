# import the necessary packages
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import glob
import json
import numpy as np

class TrainingStatusMonitor(Callback):
    def __init__(self, jsonStatusOutputPath, jsonMetricsPath=None, modelCheckPointCallbacks = None, startAt=0):
        # call the parent constructor
        super(TrainingStatusMonitor, self).__init__()
        
        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.jsonStatusOutputPath = jsonStatusOutputPath
        self.jsonMetricsPath = jsonMetricsPath
        self.modelCheckPointCallbacks = modelCheckPointCallbacks
        self.startAt = startAt
        self.intEpoch = startAt

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.TrainingStatus = {}

        # if the JSON history path exists, load the training history
        if self.jsonMetricsPath is not None:
            if os.path.exists(self.jsonMetricsPath):
                H = json.loads(open(self.jsonMetricsPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    ranges = [idx for idx in range(len(H["epoch"])) if (H["epoch"][idx]>=0 and H["epoch"][idx]<self.startAt+1)]
                    if (ranges != []):
                        startAtIndex = np.max(ranges)
                        for k in H.keys():
                            H[k] = H[k][:startAtIndex+1]
                        self.intEpoch = H["epoch"][-1] 
                        self.TrainingStatus["current_epoch"] = self.intEpoch

                # get min, max value from metrics history
                self.get_status_from_history(H)
        
                # update best value for callbacks
                if self.modelCheckPointCallbacks is not None:
                    for callback in self.modelCheckPointCallbacks:
                        if isinstance(callback, ModelCheckpoint):
                            monitor = callback.monitor
                            if callback.monitor_op is np.less:
                                key = "min_" + monitor
                            elif callback.monitor_op is np.greater:
                                key = "max_" + monitor
                            callback.best = self.TrainingStatus[key]
    
    def on_epoch_end(self, epoch, logs={}):
        # save current epoch
        self.TrainingStatus["current_epoch"] = self.intEpoch + 1

        H = {}
        if self.jsonMetricsPath is not None:
            if os.path.exists(self.jsonMetricsPath):
                H = json.loads(open(self.jsonMetricsPath).read())

        # get min, max value from metrics history
        self.get_status_from_history(H)

        # check to see if the training history should be serialized
        # to file
        if self.jsonStatusOutputPath is not None:
            f = open(self.jsonStatusOutputPath, "w")
            f.write(json.dumps(self.TrainingStatus))
            f.close()
           
        # increment the internal epoch counter
        self.intEpoch += 1

    def get_status_from_history(self, H):
        for (k,v) in H.items():
            if (k != "epoch"):
                max_index = np.nanargmax(v)
                min_index = np.nanargmin(v)
                max_epoch = H['epoch'][max_index]
                max_value = H[k][max_index]
                min_epoch = H['epoch'][min_index]
                min_value = H[k][min_index]
                self.TrainingStatus["max_epoch_"+k] = max_epoch
                self.TrainingStatus["min_epoch_"+k] = min_epoch
                self.TrainingStatus["max_"+k] = max_value
                self.TrainingStatus["min_"+k] = min_value
