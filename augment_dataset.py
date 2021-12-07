import os
import sys

import skimage
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
os.chdir(__location__)

import argparse
import traceback
from configure import config_training
from util import file_methods, url_methods
from preprocessing.geometric_transforming import GeometricTransforming
from augmenting.image_mesh_augmenting import ImageMeshAugmenting
from preprocessing.image3D_extracting import Image3DExtracting
from tqdm import tqdm
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-zipd", "--zipDir", type=str, 
    default=r"E:\My Drive\CaoHoc\LUANVAN\SourceCode\data",
    help="path to download zip dataset")
ap.add_argument("-ed", "--extractdir", type=str,
    default=r"D:\Study\CaoHoc\LUANVAN\Dataset",
    help="path to extract dataset")
ap.add_argument("-nd", "--numdataset", type=int,
    default=0,
    help="specific number of dataset for running range 0 to {}".format(len(config_training.DATASET_DIRS)-1))
ap.add_argument("-o", "--outputdir", type=str,
    default=r"D:\Study\CaoHoc\LuanVan\dataset",
    help="path to extract dataset")
ap.add_argument("-pf", "--posfix", type=str,
    default=r"_augment_6000",
    help="path to extract dataset")
ap.add_argument("-ts", "--totalsize", type=int,
    #default=30613, # val 20%
	# default=13000, # val 10#
    default=8000,
    help="path to extract dataset")
args = vars(ap.parse_args())

DATASET_EXTRACT_PATH = args["extractdir"]
DATA_ZIP_PATH = args["zipDir"]
NUM_DATASET = args["numdataset"]
TOTAL_SIZE = args["totalsize"]
POSFIX = args["posfix"]

datasetURLs = config_training.DATASET_URLS[:]
datasetZipNames = config_training.DATASET_ZIP_NAMES[:]
datasetDirs = config_training.DATASET_DIRS[:]
if (NUM_DATASET >= 0 and NUM_DATASET < len(datasetDirs)):
    datasetDirs = [datasetDirs[NUM_DATASET]]
    datasetZipNames = [datasetZipNames[NUM_DATASET]]
    datasetURLs = [datasetURLs[NUM_DATASET]]

print("[INFO] Loading image and mat paths from {}...".format(DATASET_EXTRACT_PATH))
if os.path.exists(DATASET_EXTRACT_PATH) is False:
    os.mkdir(DATASET_EXTRACT_PATH)
extractedFilePaths = []
for (url, datasetDir, zipName) in zip(datasetURLs, datasetDirs, datasetZipNames):

    zipFilePath = os.path.sep.join([DATA_ZIP_PATH, zipName])
    extractedFilePath = os.path.sep.join([DATASET_EXTRACT_PATH, datasetDir])
    extractedToFilePath = os.path.sep.join([DATASET_EXTRACT_PATH, 
        file_methods.getFileNameWithoutExt(zipName), datasetDir])
    
    if (os.path.exists(extractedFilePath) is False and
        os.path.exists(extractedToFilePath) is False):
        print("[INFO] Downloading and unzip the dataset {}...".format(datasetDir))
        try:
            if (os.path.exists(zipFilePath) is False):
                url_methods.download(url, DATASET_EXTRACT_PATH)
            url_methods.unzip(zipFilePath, DATASET_EXTRACT_PATH)
        except Exception as e:
            print("[ERROR] Occuring error when download and unzip the dataset {}".format(datasetDir))
            print(traceback.format_exc())
            continue

    if (os.path.exists(extractedFilePath) is True):
        extractedFilePaths.append(extractedFilePath)
    elif (os.path.exists(extractedToFilePath) is True):
        extractedFilePaths.append(extractedToFilePath)

if len(extractedFilePaths) == 0:
    print("[ERROR] No dataset for augmenting")
    exit()

print("[INFO] Augmenting image and mat for datasets...")
OUTPUT_DIR = args["outputdir"]
gt = GeometricTransforming()
i3e = Image3DExtracting()
ima = ImageMeshAugmenting()
for i, extractedFilePath in enumerate(extractedFilePaths):
    np.random.seed(1189 + i)

    datasetName = datasetDirs[i]

    print("[INFO] Processing for dataset {}...".format(datasetName))
    ouputDirName = datasetName + POSFIX
    ouputDirPath = file_methods.makeDirs(ouputDirName, OUTPUT_DIR)

    print("Loading image and mat paths...")
    imageWithMatPaths = file_methods.getImageWithMatList(extractedFilePath)
    totalAugumentSize = TOTAL_SIZE - len(imageWithMatPaths)
    augumentNumEachImage = int(np.ceil(totalAugumentSize/len(imageWithMatPaths)))
    if totalAugumentSize <= 0:
        print("Not continue because total size less than size of original: {} < {}".format(
            TOTAL_SIZE, len(imageWithMatPaths)
        ))
        continue
    else:
        print("Augument info: augument for each image (imcluding flip): {}".format(augumentNumEachImage))

    print("Creating agument image dirs...")
    ouputDirOrgPath = ouputDirPath

    print("Creating agument images with original at {}...".format(ouputDirOrgPath))
    pbar = tqdm(desc="Generating images", total=len(imageWithMatPaths)*augumentNumEachImage)
    ima.borderMode="constant"
    for (imagePath, matPath) in imageWithMatPaths:
        image = file_methods.readImage(imagePath)
        (h, w) = (image.shape[0], image.shape[1])
        mat = file_methods.readMat(matPath)
        
        meshInfo = i3e.preprocess(matPath)

        for i in range(augumentNumEachImage):
            posfix = "{:04d}".format(i)
            augImagePath = file_methods.getFilePathWithOtherParent(imagePath, ouputDirOrgPath, posfix)
            augMatPath = file_methods.getFilePathWithOtherParent(matPath, ouputDirOrgPath, posfix)

            (augImage, augMeshInfo, augTform) = ima.augment(image, meshInfo)
            augTformDict = { "tform": augTform.params,
                "image_size": [h, w] }
            file_methods.saveImage(augImagePath, skimage.img_as_ubyte(augImage))
            file_methods.saveMat(augMatPath, mat, augTformDict)

            pbar.update(1)

            # from preprocessing.image3D_to_2D import Image3DTo2D
            # i3t2 = Image3DTo2D()
            # reloadAugImage = file_methods.readImage(augImagePath)
            # reloadAugMeshInfo = i3e.preprocess(augMatPath)
            # reloadAugImageFace = i3t2.preprocess(reloadAugMeshInfo, h, w, False)
            # reloadAugImageFaceKpts = i3t2.drawKeypoints(reloadAugImageFace, reloadAugMeshInfo)

            # import matplotlib.pyplot as plt
            # plt.subplot(2, 1, 1)
            # stackImage = np.concatenate((image, reloadAugImage, reloadAugImageFaceKpts), axis=1)
            # plt.imshow(stackImage)
            # plt.subplot(2, 1, 2)
            # plt.imshow(reloadAugImage)
            # plt.imshow(reloadAugImageFaceKpts, alpha=.5)
            # plt.show(block=True)
