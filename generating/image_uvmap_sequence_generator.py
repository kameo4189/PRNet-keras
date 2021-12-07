import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
from tensorflow.keras.utils import Sequence
from preprocessing.image3D_extracting import Image3DExtracting
from preprocessing.uvmap_meshinfo_converting import UVMapMeshInfoConverting
from preprocessing.image_mesh_warping import ImageMeshWarping
from preprocessing.uvmap_creating import UVMapCreating
from inout.hdf5datasetreader import HDF5DatasetReader
from util import file_methods
import skimage
import scipy.io as sio
from configure import config
import warnings

class ImageUVMapSequenceGenerator(Sequence):
    passes = np.inf
    i3e = Image3DExtracting()
    imw = ImageMeshWarping()
    uvc = UVMapCreating()
    umc = UVMapMeshInfoConverting()

    def __init__(self, hdf5Path, bs, initEpoch, initRandomSeed,
        aug=None, shuffle=True, dataTypeMode=config.DATA_TYPE_MODE_DEFAULT, mode="train",
        preload=True, useImageNum=-1, generateMode=1):
        assert bs > 0, "Batch size is negative"

        # 0: augment before warp, 1: warp before augment
        self.generateMode = generateMode

        self.bs = bs
        self.epochs = initEpoch
        self.initRandomSeed = initRandomSeed
        self.aug = aug
        self.randomPertubeBB = False if self.aug is None else True
        self.shuffle = shuffle
        self.dataTypeMode = dataTypeMode
        self.mode = mode
        self.uvMaxPos = max(self.uvc.uvHeight, self.uvc.uvWidth) * config.UV_POS_SCALE

        self.dbReader = HDF5DatasetReader(hdf5Path)
        self.dbImage = self.dbReader.image
        self.dbUVMap = self.dbReader.uvmap
        self.dbImageDType = self.dbImage.dtype
        self.dbUVMapDType = self.dbUVMap.dtype
        self.dbImageShape = self.dbImage.shape[1:]
        self.dbUVMapShape = self.dbUVMap.shape[1:]
        self.UVMapShape = (config.UV_HEIGHT, config.UV_HEIGHT, config.COLOR_CHANNEL)

        self.numImages = self.dbReader.totalSize
        if useImageNum > 0 and useImageNum < self.numImages:
            self.numImages = useImageNum

        self.dataTypeMode = dataTypeMode
        if self.shuffle is False:
            np.random.seed(self.initRandomSeed)
            self.indexes = np.arange(self.numImages)
            np.random.shuffle(self.indexes)

        self.rawImages = None
        self.rawMats = None
        if preload:
            bs = 50000
            self.rawImages = []
            for i in range(0, self.numImages, bs):
                self.rawImages.extend([bytes(image) for image in self.dbImage[i:i+bs]])

            if (len(self.dbUVMap.shape) == 2):
                sizes = self.dbReader.uvmapSize[:]
                bs = 50000
                self.rawMats = []
                for i in range(0, self.numImages, bs):
                    rawMats = [bytes(mat)[:size] for (mat, size) in zip(self.dbUVMap[i:i+bs], sizes[i:i+bs])]
                    self.rawMats.extend([file_methods.readMat(mat) for mat in rawMats])

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.numImages / self.bs))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.bs:(index+1)*self.bs].tolist()
        batchIndexes = range(len(indexes))
        (indexes, batchIndexes) = zip(*sorted(zip(indexes, batchIndexes)))
        (indexes, batchIndexes) = (list(indexes), list(batchIndexes))

        # extract the images from the HDF5 dataset
        if self.rawImages is not None:
            images = [self.rawImages[i] for i in indexes]
            images = [images[i] for i in batchIndexes]
            images = [file_methods.readImage(image, True) for image in images]
        elif (len(self.dbUVMap.shape) == 2):
            images = [bytes(image) for image in self.dbImage[indexes]]
            images = [images[i] for i in batchIndexes]
            images = [file_methods.readImage(image, True) for image in images]
        else:
            images = np.empty((len(indexes), *self.dbImageShape), self.dbImageDType)
            images[batchIndexes] = self.dbImage[indexes]

        # extract the uvmaps from the HDF5 dataset
        if self.rawMats is not None:
            mats = [self.rawMats[i] for i in indexes]
            mats = [mats[i] for i in batchIndexes]
            meshInfos = [self.i3e.preprocess(mat) for mat in mats]
        elif (len(self.dbUVMap.shape) == 2):
            mats = [bytes(mat)[:size] for (mat, size) in zip(self.dbUVMap[indexes], self.dbReader.uvmapSize[indexes])]
            mats = [mats[i] for i in batchIndexes]
            mats = [file_methods.readMat(mat) for mat in mats]
            meshInfos = [self.i3e.preprocess(mat) for mat in mats]
        else:
            uvmaps = np.empty((len(indexes), *self.dbUVMapShape), self.dbUVMapDType)       
            uvmaps[batchIndexes] = self.dbUVMap[indexes]
            meshInfos = [self.umc.preprocess(image, uvmap, None) for (image, uvmap) in zip(images, uvmaps)]

        generatedData = self.__data_generation(images, meshInfos)

        return generatedData

    def on_epoch_end(self):
        self.epochs += 1
        if self.mode != "val":
            np.random.seed(self.initRandomSeed + self.epochs)
            if self.shuffle == True:
                self.indexes = np.arange(self.numImages)
                np.random.shuffle(self.indexes)

    def __data_generation(self, images, meshInfos):
        (finalImages, finaMeshInfos, finalTforms) = (images, meshInfos, None)
        (warpedFaceImages, warpedMeshInfos, warpedTforms)= (images, meshInfos, None)
        (augImages, augMeshInfos, augTforms) = (images, meshInfos, None)
        if self.generateMode == 0:
            # if the data augmentation object is not None, apply it
            (augImages, augMeshInfos, augTforms) = (images, meshInfos, None)
            if self.aug is not None:
                (augImages, augMeshInfos, augTforms) = self.aug.augments(images, meshInfos)

            # warping images and meshs
            (warpedFaceImages, warpedMeshInfos, warpedTforms) = self.imw.preprocesses(augImages,
                augMeshInfos, randomPertubeBB=self.randomPertubeBB)

            finalTforms = warpedTforms
            if augTforms is not None:
                finalTforms = [augTform + warpedTform for (augTform, warpedTform) in zip(augTforms, warpedTforms)]

            (finalImages, finaMeshInfos, finalTforms) = (warpedFaceImages, warpedMeshInfos, finalTforms)
        else:
            # warping images and meshs
            (warpedFaceImages, warpedMeshInfos, warpedTforms) = self.imw.preprocesses(images,
                meshInfos, randomPertubeBB=self.randomPertubeBB)

            (finalImages, finaMeshInfos, finalTforms) = (warpedFaceImages, warpedMeshInfos, warpedTforms)
            if self.aug is not None:
                (augImages, augMeshInfos, augTforms) = self.aug.augments(warpedFaceImages, warpedMeshInfos)
                finalTforms = [warpedTform + augTform for (warpedTform, augTform) in zip(warpedTforms, augTforms)]
                (finalImages, finaMeshInfos, finalTforms) = (augImages, augMeshInfos, finalTforms)

        # creating uv position map
        uvPosMaps = [self.uvc.preprocess(meshInfo, type="pos") for meshInfo in finaMeshInfos]
        uvPosMaps = [uvPosMap/self.uvMaxPos for uvPosMap in uvPosMaps]

        # convert face image to uint8 and uvmap to float32
        imageDType = config.DATA_TYPE_MODES[self.dataTypeMode][0]
        uvmapDType = config.DATA_TYPE_MODES[self.dataTypeMode][1]
        if (imageDType == "uint8"):
            warnings.filterwarnings("ignore", category=UserWarning)
            faceImages = [skimage.img_as_ubyte(faceImage) for faceImage in finalImages]
            warnings.filterwarnings("default", category=UserWarning)
        else:
            faceImages = [faceImage.astype(imageDType) for faceImage in finalImages]
        uvPosMaps = [uvPosMap.astype(uvmapDType) for uvPosMap in uvPosMaps]

        # yield a tuple of images and labels
        if self.mode == "dev":
            return (images, warpedFaceImages, faceImages, uvPosMaps, finalTforms)
        else:
            return (np.array(faceImages), np.array(uvPosMaps))

if __name__ == "__main__":
    import numpy as np
    import scipy.io as sio
    from skimage import io
    import matplotlib.pyplot as plt
    from util import file_methods
    from augmenting.image_mesh_augmenting import ImageMeshAugmenting
    from postprocessing.uvmap_restoring import UVMapRestoring
    from preprocessing.image3D_to_2D import Image3DTo2D
    import skimage.transform as skt
    from tensorflow.python.keras.utils import data_utils
    from util import mesh_display

    ima = ImageMeshAugmenting()
    uvr = UVMapRestoring()
    i3t2 = Image3DTo2D()
    bs = 4

    hdf5Path = r"D:\Study\CaoHoc\LUANVAN\HDF5\train.hdf5"
    imdg = ImageUVMapSequenceGenerator(hdf5Path, bs, 0, 1189, aug=ima, mode="dev", preload=False, generateMode=1)
    generatorFunc = data_utils.iter_sequence_infinite(imdg)

    while True:
        print("[INFO] generating images and uvmaps...")
        generatedItems = next(generatorFunc)
        generatedItems = list(zip(*generatedItems))

        print("[INFO] show images and maps...")
        plt.figure(figsize=(50,150))
        for i, item in enumerate(generatedItems):
            image, warpedFaceImage, faceImage, uvPosMap, tform = item

            hOrg, wOrg = image.shape[0], image.shape[1]

            restoredMeshInfoOrg = uvr.postprocess(image, uvPosMap, tform)
            restoreMeshFaceOrg = i3t2.preprocess(restoredMeshInfoOrg, hOrg, wOrg, False)
            restoreMeshFaceOrgKpts = i3t2.drawKeypoints(restoreMeshFaceOrg, restoredMeshInfoOrg)

            h, w = faceImage.shape[0], faceImage.shape[1]

            restoreMeshInfo = uvr.postprocess(faceImage, uvPosMap, None)
            
            restoreMeshFace = i3t2.preprocess(restoreMeshInfo, h, w, False)
            restoreMeshFaceKpts = i3t2.drawKeypoints(restoreMeshFace, restoreMeshInfo)         

            plt.subplot(bs,2,(i*2)+1)
            stackImage = np.concatenate((uvPosMap, warpedFaceImage, faceImage, restoreMeshFaceKpts), axis=1)
            plt.imshow(stackImage)

            plt.subplot(bs,2,(i*2)+2)
            stackImageOrg = np.concatenate((image, restoreMeshFaceOrgKpts), axis=1)
            plt.imshow(stackImageOrg)

            mesh_display.displayPointCloudColor(restoreMeshInfo)

        plt.show(block=True)



        


