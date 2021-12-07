# import the necessary packages
from tensorflow.keras.callbacks import Callback
import os
import numpy as np
import shutil
import glob
from IPython.display import FileLink

class TrainingOutputZipping(Callback):
    def __init__(self, modelOutputPath, zipOutputPath, every=1, startAt=0, max_to_keep=5):
        # call the parent constructor
        super(TrainingOutputZipping, self).__init__()
        
        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.modelOutputPath = modelOutputPath
        self.zipOutputPath = zipOutputPath
        self.every = every
        self.intEpoch = startAt
        self.max_to_keep = max_to_keep
    
    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model output should be serialized to disk
        if (self.intEpoch+1) % self.every == 0:
            p = os.path.sep.join([self.zipOutputPath,
                "epoch_{:04d}".format(self.intEpoch+1)])
            shutil.make_archive(p, 'zip', self.modelOutputPath)   
            print("Epoch {:05d}: output of model was archived as link below:".format(self.intEpoch+1))
            print(FileLink(p + ".zip"))

        # only keep last n checkpoints
        checkpointPaths = sorted(glob.glob(os.sep.join([self.zipOutputPath, "epoch_*.zip"])), reverse = True)
        for i, checkpointPath in enumerate(checkpointPaths):
            if i >= self.max_to_keep:
                os.remove(checkpointPath)
            
        # increment the internal epoch counter
        self.intEpoch += self.every
