# import the necessary packages
import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from tqdm import tqdm
from generating.image_uvmap_generator import ImageUVMapGenerator
from util import file_methods
from configure import config
import numpy as np

from multiprocessing import Pool
from itertools import repeat
from augmenting.image_mesh_augmenting import ImageMeshAugmenting

imdg = ImageUVMapGenerator()
aug = ImageMeshAugmenting()
def generate_item_aug(imagePath, matPath, dataTypeMode):
    faceImage, uvPosMap = imdg.generate(imagePath, matPath, aug, dataTypeMode)
    return faceImage, uvPosMap

def generate_item_noaug(imagePath, matPath, dataTypeMode):
    faceImage, uvPosMap = imdg.generate(imagePath, matPath, None, dataTypeMode)
    return faceImage, uvPosMap

class DatasetLoader:
    imdg = ImageUVMapGenerator()

    def load(self, imagePaths, matPaths, aug=None, dataTypeMode=config.DATA_TYPE_MODE_DEFAULT,
        warping=True):
        # initialize the list of images and mats
        images = []
        uvmaps = []

        # loop over the input images
        for (imagePath, matPath) in zip(imagePaths, matPaths):
            faceImage, uvPosMap = self.imdg.generate(imagePath, matPath, aug, dataTypeMode, warping)

            # updating the data list followed by the mats
            images.append(faceImage)
            uvmaps.append(uvPosMap)

        return images, uvmaps

    def load_multiprocessing(self, imagePaths, matPaths, isAug, numThread, dataTypeMode):        
        with Pool(processes=numThread) as pool:
            if (isAug):
                images, uvmaps = zip(*pool.starmap(generate_item_aug, zip(imagePaths, matPaths, repeat(dataTypeMode, len(imagePaths)))))
            else:
                images, uvmaps = zip(*pool.starmap(generate_item_noaug, zip(imagePaths, matPaths, repeat(dataTypeMode, len(imagePaths)))))
        return images, uvmaps