# import the necessary packages
# import the necessary packages
import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from configure import config
import numpy as np
import h5py
import skimage
from inout.hdf5datasetreader import HDF5DatasetReader
from util import file_methods

class HDF5DatasetsGenerator:
    def __init__(self, dbDir, prefix, batchSize, dbUseNum=0):
        self.batchSize = batchSize

        # get all HDF5 databases
        FILE_EXT = ".hdf5"
        dbPaths = file_methods.getAllFiles(dbDir, FILE_EXT)
        dbPaths = [file for file in dbPaths if file_methods.getFileName(file).startswith(prefix)]
        fileIndexes = [file_methods.getFileNameWithoutExt(x).split('_')[-1] for x in dbPaths] 
        fileIndexes = [int(x) if x.isnumeric() == True else 0 for x in fileIndexes]
        dbPaths = [x for _, x in sorted(zip(fileIndexes, dbPaths))]
        if dbUseNum > 0:
            dbPaths = dbPaths[:dbUseNum]

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.dbReaders = [HDF5DatasetReader(dbPath) for dbPath in dbPaths]
        self.dbDataSizes = [dbReader.totalSize for dbReader in self.dbReaders]
        self.dbImages = [dbReader.db["image"] for dbReader in self.dbReaders]
        self.dbUVMaps = [dbReader.db["uvmap"] for dbReader in self.dbReaders]
        self.dbImageDType = self.dbImages[0].dtype
        self.dbUVMapDType = self.dbUVMaps[0].dtype
        self.dbImageShape = self.dbImages[0].shape[1:]
        self.dbUVMapShape = self.dbUVMaps[0].shape[1:]

        self.numImages = sum(self.dbDataSizes)
        self.dbDataRanges = [sum(self.dbDataSizes[:i]) for i in range(len(self.dbDataSizes))]
        self.dbDataRanges.append(self.numImages)
        self.dbDataRanges = [(self.dbDataRanges[i], self.dbDataRanges[i+1]) for i in range(len(self.dbDataRanges)-1)]
        
        self.indexes = list(range(self.numImages))        
        np.random.shuffle(self.indexes)
        self.dbReaderIndexes = [self.toDBReaderIndex(index) for index in self.indexes]
        self.dbReaderDataIndexes = [self.toBDReaderDataIndex(index) for index in self.indexes]

        self.batchIndexDicts = []
        for i in np.arange(0, self.numImages, self.batchSize):
            indexes = self.indexes[i: i + self.batchSize]
            batchIndexes = list(range(len(indexes)))
            dbReaderIndexes = self.dbReaderIndexes[i: i + self.batchSize]
            dbReaderDataIndexes = self.dbReaderDataIndexes[i: i + self.batchSize]
            combinedIndexes = zip(dbReaderIndexes, batchIndexes, dbReaderDataIndexes)

            batchIndexDict = {}
            for (dbReaderIndex, batchIndex, dbReaderDataIndex) in combinedIndexes:
                batchIndexItems = batchIndexDict.get(dbReaderIndex, [])
                batchIndexItems.append([dbReaderDataIndex, batchIndex])
                batchIndexDict[dbReaderIndex] = batchIndexItems

            for (key, item) in batchIndexDict.items():
                batchIndexDict[key] = sorted(item)

            self.batchIndexDicts.append((batchIndexDict, len(indexes)))
    
    def toDBReaderIndex(self, imageIndex):
        dbReaderIndexes = [dbIndex for dbIndex in range(len(self.dbDataRanges)) 
            if imageIndex >= self.dbDataRanges[dbIndex][0] and imageIndex < self.dbDataRanges[dbIndex][1]]
        return dbReaderIndexes[0]
    
    def toBDReaderDataIndex(self, imageIndex):
        dbReaderDataIndexes = [imageIndex - self.dbDataRanges[dbIndex][0] for dbIndex in range(len(self.dbDataRanges)) 
            if imageIndex >= self.dbDataRanges[dbIndex][0] and imageIndex < self.dbDataRanges[dbIndex][1]]
        return dbReaderDataIndexes[0]

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over pre-arange batch indexes
            for (batchIndexDict, length) in self.batchIndexDicts:
                images = np.empty((length, *self.dbImageShape), self.dbImageDType)
                uvmaps = np.empty((length, *self.dbUVMapShape), self.dbUVMapDType)

                for (key, item) in batchIndexDict.items():
                    (dbDataIndexes, batchIndexes) = zip(*item)
                    (dbDataIndexes, batchIndexes) = (list(dbDataIndexes), list(batchIndexes))
                    images[batchIndexes] = self.dbImages[key][dbDataIndexes]
                    uvmaps[batchIndexes] = self.dbUVMaps[key][dbDataIndexes]

                if self.dbImageDType is np.uint8:
                    images = [skimage.img_as_float(image) for image in images]

                # yield a tuple of images and uvmaps
                yield np.array(images), np.array(uvmaps)

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        [dbReader.close() for dbReader in self.dbReaders]

if __name__ == "__main__":
    import argparse
    from configure import config_training
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-pre", "--prefix", type=str, 
        default=file_methods.getFileNameWithoutExt(config_training.TRAIN_HDF5_NAME),
        help="base file name")
    ap.add_argument("-hd", "--hdf5dir", type=str,
        # default=r"D:\GoogleDrive\CaoHoc\LUANVAN\SourceCode\data\hdf5",
        default=r"K:\Study\CaoHoc\LuanVan\Dataset\hdf5",
        help="path to hdf5 datasets")
    args = vars(ap.parse_args())

    FILE_PREFIX = args["prefix"]
    HDF5_DIR = args["hdf5dir"]

    from configure import config
    from configure import config_training    
    import matplotlib.pyplot as plt
    from postprocessing.uvmap_restoring import UVMapRestoring
    # from util import mesh_display
    from preprocessing.image3D_to_2D import Image3DTo2D

    # files = file_methods.getAllFiles(HDF5_DIR, FILE_EXT)
    # files = [file for file in files if file_methods.getFileName(file).startswith(FILE_PREFIX)]
    # fileIndexes = [file_methods.getFileNameWithoutExt(x).split('_')[-1] for x in files] 
    # fileIndexes = [int(x) if x.isnumeric() == True else 0 for x in fileIndexes]
    # files = [x for _, x in sorted(zip(fileIndexes, files))]

    #TRAIN_HDF5 = os.path.sep.join([r"D:\GoogleDrive\CaoHoc\LUANVAN\SourceCode\data\hdf5", config_training.TRAIN_HDF5_NAME])
    # TRAIN_HDF5 = os.path.sep.join([r"D:\Study\CaoHoc\LUANVAN\hdf5", config_training.TRAIN_HDF5_NAME])
    trainGen = HDF5DatasetsGenerator(HDF5_DIR, FILE_PREFIX, 16)
    trainGenFunc = trainGen.generator()

    uvr = UVMapRestoring()
    i3t2 = Image3DTo2D()

    while True:
        print("[INFO] generating images and maps...")
        generatedItems = next(trainGenFunc)
        generatedItems = list(zip(*generatedItems))

        print("[INFO] show images and maps...")
        plt.figure(figsize=(50,100)) 
        for i, item in enumerate(generatedItems):
            faceImage, uvmPosMap = item

            h, w = faceImage.shape[0], faceImage.shape[1]
            restoreMeshInfo = uvr.postprocess(faceImage, uvmPosMap, None)

            # restoreMeshFace = i3t2.preprocess(restoreMeshInfo, h, w, False)
            # restoreMeshFaceKpts = i3t2.drawKeypoints(restoreMeshFace, restoreMeshInfo)

            plt.subplot(len(generatedItems)/4, 4, i+1)
            # stackImage = np.concatenate((uvmPosMap/256., faceImage, restoreMeshFaceKpts), axis=1)
            stackImage = np.concatenate((uvmPosMap/256., faceImage), axis=1)
            plt.imshow(stackImage)

        plt.show()