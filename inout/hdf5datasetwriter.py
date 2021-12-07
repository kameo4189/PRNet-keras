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

class HDF5DatasetWriter:
    groupNumberKey = "groupNumber"
    groupLabelKey = "groupLabel"
    uvmapSizeKey = "uvmapSize"

    def __init__(self, outputPath, dims=None, uvmapDims=None, imageKey="image", imageDType='byte',
        uvmapKey="uvmap", uvmapDType='float16', saveUVMapSize=False, bufSize=1000):
        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(outputPath):
            print ("[INFO] The supplied <outputPath> already exists, it is copied to <outputPath>_<current_time>")
            self.backupDataset(outputPath)

        self.createDataset(dims, uvmapDims, outputPath, imageKey, imageDType, uvmapKey, uvmapDType, saveUVMapSize)
        self.db.flush()
        
        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"image": [], "uvmap": [], self.uvmapSizeKey: []}

    def createDataset(self, dims, uvmapDims, outputPath, imageKey, imageDType, uvmapKey, uvmapDType, saveUVMapSize):
        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class masks
        self.db = h5py.File(outputPath, "w")
        self.image = self.db.create_dataset(imageKey, dims, dtype=imageDType)
        self.uvmap = self.db.create_dataset(uvmapKey, uvmapDims, dtype=uvmapDType)
        self.uvMapSize = None
        if saveUVMapSize:
            self.uvMapSize = self.db.create_dataset(self.uvmapSizeKey, (uvmapDims[0],), dtype="uint")    
        self.idx = 0            

    def backupDataset(self, outputPath):        
        dir = file_methods.getParentPath(outputPath)
        ext = file_methods.getExt(outputPath)
        name = file_methods.getFileNameWithoutExt(outputPath)
        new_name = name + "_" + strftime("%m_%d_%Y_%H_%M_%S") + ext
        new_path = os.path.sep.join([dir, new_name])         
        copyfile(outputPath, new_path)
        os.remove(outputPath)

    def add(self, rows, uvmaps, uvmapSizes=None, dbFlush=True):
        # add the rows and masks to the buffer
        self.buffer["image"].extend(rows)
        self.buffer["uvmap"].extend(uvmaps)
        if (uvmapSizes is not None and self.uvMapSize is not None):
            self.buffer[self.uvmapSizeKey].extend(uvmapSizes)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["image"]) >= self.bufSize:
            self.flush()

    def flush(self, dbFlush=True):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["image"])
        self.image[self.idx:i] = self.buffer["image"]
        self.uvmap[self.idx:i] = self.buffer["uvmap"]
        if (self.uvMapSize is not None):
            self.uvMapSize[self.idx:i] = self.buffer[self.uvmapSizeKey]

        if dbFlush:
            self.db.flush()
        self.idx = i 
        self.buffer = {"image": [], "uvmap": [], self.uvmapSizeKey: []}

    def storeGroupNumbers(self, groupNumbers):
        if ((self.groupNumberKey in self.db.keys()) is False):
            groupNumberSet = self.db.create_dataset(self.groupNumberKey, (len(groupNumbers),), 
                dtype="int")
            groupNumberSet[:] = groupNumbers
            self.db.flush()

    def storeGroupLabels(self, groupLabels):
        if ((self.groupLabelKey in self.db.keys()) is False):
            groupLabelSet = self.db.create_dataset(self.groupLabelKey, (len(groupLabels),), 
                dtype=h5py.special_dtype(vlen=str))
            groupLabelSet[:] = groupLabels
            self.db.flush()

    def getTotalDataSize(self):
        return self.image.shape[0]

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["image"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()