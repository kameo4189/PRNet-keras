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
from preprocessing.image3D_extracting import Image3DExtracting
from generating.image_uvmap_generator import UVMapCreating
from inout.hdf5datasetreader import HDF5DatasetReader
from util import file_methods
import warnings

class HDF5DatasetGenerator:
    i3e = Image3DExtracting()
    uvc = UVMapCreating()

    def __init__(self, dbPath, batchSize, dataTypeMode=config.DATA_TYPE_MODE_DEFAULT):
        self.batchSize = batchSize

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.dbReader = HDF5DatasetReader(dbPath)
        self.dbImage = self.dbReader.image
        self.dbUVMap = self.dbReader.uvmap
        self.dbUVMapSize = self.dbReader.uvmapSize
        self.dbImageDType = self.dbImage.dtype
        self.dbUVMapDType = self.dbUVMap.dtype
        self.dbImageShape = self.dbImage.shape[1:]
        self.dbUVMapShape = self.dbUVMap.shape[1:]

        self.UVMapShape = (config.UV_HEIGHT, config.UV_HEIGHT, config.COLOR_CHANNEL)

        self.numImages = self.dbReader.totalSize
        self.indexes = list(range(self.numImages))
        self.dataTypeMode = dataTypeMode
        np.random.shuffle(self.indexes)

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0       
        imageDType = config.DATA_TYPE_MODES[self.dataTypeMode][0]
        uvmapDType = config.DATA_TYPE_MODES[self.dataTypeMode][1]

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batchSize):
                indexes = self.indexes[i: i + self.batchSize]
                batchIndexes = range(len(indexes))
                (indexes, batchIndexes) = zip(*sorted(zip(indexes, batchIndexes)))
                (indexes, batchIndexes) = (list(indexes), list(batchIndexes))

                # extract the images and uvmaps from the HDF dataset
                images = np.empty((len(indexes), *self.dbImageShape), self.dbImageDType)
                images[batchIndexes] = self.dbImage[indexes]
                if (len(self.dbImage.shape) == 2):
                    images = [bytes(image) for image in images]
                    images = [file_methods.readImage(image, True) for image in images]

                uvmaps = np.empty((len(indexes), *self.dbUVMapShape), self.dbUVMapDType)
                dbUVMaps = self.dbUVMap[indexes]
                if (len(self.dbUVMap.shape) == 2):
                    uvmaps = np.empty((len(indexes), *self.UVMapShape), uvmapDType)

                    sizes = self.dbUVMapSize[indexes]
                    mats = [bytes(mat)[:size] for (mat, size) in zip(dbUVMaps, sizes)]
                    mats = [file_methods.readMat(mat) for mat in mats]
                    meshInfos = [self.i3e.preprocess(mat) for mat in mats]
                    dbUVMaps = [self.uvc.preprocess(meshInfo, type="pos",
                        h=image.shape[0], w=image.shape[1])
                        for (image, meshInfo) in zip(images, meshInfos)]
                    dbUVMaps = np.array(dbUVMaps)
                uvmaps[batchIndexes] = dbUVMaps

                if (imageDType == "uint8"):
                    warnings.filterwarnings("ignore", category=UserWarning)
                    images = [skimage.img_as_ubyte(image) for image in images]
                    warnings.filterwarnings("default", category=UserWarning)
                else:
                    images = [image.astype(imageDType) for image in images]
                uvmaps = [uvmap.astype(uvmapDType) for uvmap in uvmaps]
            
                # yield a tuple of images and uvmaps
                yield (np.array(images), np.array(uvmaps))

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.dbReader.close()

if __name__ == "__main__":
    from configure import config_training
    import matplotlib.pyplot as plt
    from postprocessing.uvmap_restoring import UVMapRestoring
    from util import mesh_display
    from preprocessing.image3D_to_2D import Image3DTo2D

    # TRAIN_HDF5 = os.path.sep.join([r"D:\GoogleDrive\CaoHoc\LUANVAN\SourceCode\data\hdf5", config_training.TRAIN_HDF5_NAME])
    # TRAIN_HDF5 = os.path.sep.join([r"D:\Study\CaoHoc\LUANVAN\hdf5", config_training.TRAIN_HDF5_NAME])
    bs = 2
    TRAIN_HDF5 = os.path.sep.join([r"K:\Study\CaoHoc\LuanVan\raw_hdf5", config_training.TRAIN_HDF5_NAME])
    trainGen = HDF5DatasetGenerator(TRAIN_HDF5, bs)
    trainGenFunc = trainGen.generator()
    # (images, uvmaps) = next(trainGenFunc)

    uvr = UVMapRestoring()
    i3t2 = Image3DTo2D()

    while True:
        print("[INFO] generating images and maps...")
        generatedItems = next(trainGenFunc)
        generatedItems = list(zip(*generatedItems))

        print("[INFO] show images and maps...")
        plt.figure(figsize=(50,100)) 
        for i, item in enumerate(generatedItems):
            faceImage, uvPosMap = item

            h, w = faceImage.shape[0], faceImage.shape[1]
            restoreMeshInfo = uvr.postprocess(faceImage, uvPosMap, None)
            restoreMeshFace = i3t2.preprocess(restoreMeshInfo, h, w, False)
            restoreMeshFaceKpts = i3t2.drawKeypoints(restoreMeshFace, restoreMeshInfo)

            plt.subplot(len(generatedItems), 2, (i*2)+1)
            plt.imshow(uvPosMap/256.)
            plt.subplot(len(generatedItems), 2, (i*2)+2)
            stackImage = np.concatenate((faceImage, restoreMeshFaceKpts), axis=1)
            plt.imshow(stackImage)

            mesh_display.displayPointCloudColor(restoreMeshInfo)

        plt.show(block=True)