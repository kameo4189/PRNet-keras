# import the necessary packages
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from util import file_methods

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        self.intEpoch = startAt

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
                    #ranges = np.where(np.logical_and(self.H["epoch"]>=0, self.H["epoch"]<self.startAt+1))
                    ranges = [idx for idx in range(len(self.H["epoch"])) if (self.H["epoch"][idx]>=0 and self.H["epoch"][idx]<self.startAt+1)]
                    if (ranges != []):
                        startAtIndex = np.max(ranges)
                        for k in self.H.keys():
                            self.H[k] = self.H[k][:startAtIndex+1]
                        self.intEpoch = self.H["epoch"][-1]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            if k is not "lr":
                l = self.H.get(k, [])
                l.append(float(v))
                self.H[k] = l

        # append epoch num
        l = self.H.get("epoch", [])
        l.append(self.intEpoch+1)
        self.H["epoch"] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = self.H["epoch"]
            plt.style.use("ggplot")
            plt.figure()
            for (k, v) in self.H.items():
                if (k != "epoch"):
                    plt.plot(N, v, label=k)
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
                np.max(self.H["epoch"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()

            # plot the training loss
            plt.style.use("ggplot")
            plt.figure()
            for (k, v) in self.H.items():
                if (k.find("loss") != -1):
                    plt.plot(N, v, label=k)
            plt.title("Training Loss [Epoch {}]".format(np.max(self.H["epoch"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()

            # save the figure
            lossFigName = file_methods.getFileNameWithoutExt(self.figPath) + "_loss.png"
            lossFigDir = file_methods.getParentPath(self.figPath)
            lossFigPath = os.path.sep.join([lossFigDir, lossFigName])
            plt.savefig(lossFigPath)
            plt.close()

        # increment the internal epoch counter
        self.intEpoch += 1

