# import the necessary packages
import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import h5py
from time import strftime
from shutil import copyfile
from util import file_methods

class HDF5DatasetReader:
    groupNumberKey = "groupNumber"
    groupLabelKey = "groupLabel"
    uvmapSizeKey = "uvmapSize"

    def __init__(self, datasetPath, imageKey="image", uvmapKey="uvmap"):
        self.db = h5py.File(datasetPath, mode="r")
        self.image = self.db[imageKey]
        self.uvmap = self.db[uvmapKey]

        self.uvmapSize = None
        if self.uvmapSizeKey in self.db.keys():
            self.uvmapSize = self.db[self.uvmapSizeKey]

        self.groupLabels = None
        self.groupLabelList = None
        if self.groupLabelKey in self.db.keys():
            self.groupLabels = self.db[self.groupLabelKey]
            self.groupLabelList = self.groupLabels[:]

        self.groupNumbers = None
        self.groupNumbeList = None
        if self.groupNumberKey in self.db.keys():
            self.groupNumbers = self.db[self.groupNumberKey]
            self.groupNumbeList = self.groupNumbers[:]

        self.totalSize = self.image.shape[0]

    def getData(self, startIdx, endIdx):
        return self.image[startIdx:endIdx], self.uvmap[startIdx:endIdx]

    def close(self):
        # close the dataset
        self.db.close()