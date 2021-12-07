# import the necessary packages
from tensorflow.keras.callbacks import Callback
import os
import glob

class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=1, startAt=0, max_to_keep=5, save_weights=False):
        # call the parent constructor
        super(EpochCheckpoint, self).__init__()
        
        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt
        self.max_to_keep = max_to_keep
        self.save_weights = save_weights
    
    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch+1) % self.every == 0:
            p = os.path.sep.join([self.outputPath,
                "epoch_{:04d}.hdf5".format(self.intEpoch+1)])
            self.model.save(p, overwrite=True)
            if self.save_weights:
                p = os.path.sep.join([self.outputPath,
                    "weights_{:04d}.hdf5".format(self.intEpoch+1)])
                self.model.save_weights(p, overwrite=True)

        # only keep last max_to_keep checkpoints
        checkpointPaths = sorted(glob.glob(os.sep.join([self.outputPath, "epoch_*.hdf5"])), reverse = True)
        checkpointWeightPaths = sorted(glob.glob(os.sep.join([self.outputPath, "weights_*.hdf5"])), reverse = True)
        for i, checkpointPath in enumerate(checkpointPaths):
            if i >= self.max_to_keep:
                os.remove(checkpointPath)      
        for i, checkpointWeightPath in enumerate(checkpointWeightPaths):
            if i >= self.max_to_keep:
                os.remove(checkpointWeightPath)
            
        # increment the internal epoch counter
        self.intEpoch += self.every
