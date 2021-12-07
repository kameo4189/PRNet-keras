# import the necessary packages
from tensorflow.keras.callbacks import Callback
from keras import backend as K
import numpy as np
import json
import os
import matplotlib.pyplot as plt

class CustomLRScheduler(Callback):
    def __init__(self, schedule, mode, figPath=None, jsonPath=None, startAt=0, verbose=0,
        isSetLr=True):
        super(CustomLRScheduler, self).__init__()
        self.verbose = verbose
        self.schedule = schedule
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        self.step = 1
        self.batch = 1
        self.intEpoch = startAt
        self.curLr = 0
        self.isSetLr = isSetLr
        self.mode = mode
    
    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                                
                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    ranges = [idx for idx in range(len(self.H["epoch"])) if (self.H["epoch"][idx]>=0 and self.H["epoch"][idx]<self.startAt+1)]
                    if (ranges != []):
                        startAtIndex = np.max(ranges)
                        for k in self.H.keys():
                            self.H[k] = self.H[k][:startAtIndex+1]
                        if (self.mode == "batch"):
                            self.step = self.H["step"][-1]
                        self.intEpoch = self.H["epoch"][-1]
                            
    def on_epoch_begin(self, epoch, logs={}): 
        self.batch = 1

        if (self.mode == "epoch"):
            self.on_begin()

    def on_epoch_end(self, epoch, logs={}):
        if (self.mode == "epoch"):
            self.on_end()

        # increasing training epochs
        self.intEpoch += 1

    def on_batch_begin(self, batch, logs=None):
        if (self.mode == "batch"):
            self.on_begin()

    def on_begin(self):
        if self.isSetLr is True and self.schedule is not None:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            if (self.mode == "batch"):
                self.curLr = self.schedule(self.step)
            elif (self.mode == "epoch"):
                self.curLr = self.schedule(self.intEpoch)
            if not isinstance(self.curLr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                    'should be float.')
            K.set_value(self.model.optimizer.lr, self.curLr)
            if self.verbose > 0:
                if (self.mode == "batch"):
                    print('\nStep %05d (Epoch %05d, Batch %5d): LearningRateScheduler reducing learning '
                        'rate to %s.' % (self.step, self.intEpoch+1, self.batch, self.curLr))
                elif (self.mode == "epoch"):
                    print('\nEpoch %05d: LearningRateScheduler reducing learning '
                        'rate to %s.' % (self.intEpoch+1, self.curLr))
        else:
            self.curLr = K.get_value(self.model.optimizer.lr)

    def on_end(self):
        if self.mode == "batch":
            # loop over the step and learning rates
            # for the entire training process
            values = [self.intEpoch+1, self.batch, self.step, float(self.curLr)]
            keys = ["epoch", "batch", "step", "lr"]
        elif self.mode == "epoch":
            # loop over the epoch and learning rates
            # for the entire training process
            values = [self.intEpoch+1, float(self.curLr)]
            keys = ["epoch", "lr"]

        for i, k in enumerate(keys):
            l = self.H.get(k, [])
            l.append(values[i])
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        # plot the training loss and accuracy
        if self.figPath is not None:
            if self.isSetLr:
                N = self.H["step"]
            else:
                N = self.H["epoch"]
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["lr"], label="learning rate")
            if self.mode == "batch":
                plt.title("Learning rates [Epoch {}/Batch {}/Step {}]".format(
                    self.H["epoch"][-1], self.H["batch"][-1], self.H["step"][-1]))
                plt.xlabel("Step # or Epoch #")
            elif self.mode == "epoch":
                plt.title("Learning rates [Epoch {}]".format(self.H["epoch"][-1]))
                plt.xlabel("Epoch #")
            
            plt.ylabel("Learning rates")
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()

    def on_batch_end(self, batch, logs=None):
        if (self.mode == "batch"):
            self.on_end()
        
        # increasing training batch and steps
        self.step += 1
        self.batch += 1
