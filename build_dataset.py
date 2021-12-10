import os
import sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
os.chdir(__location__)

import argparse
import csv
import traceback
from os import path
import numpy as np
from sklearn.utils import shuffle
from configure import config
from configure import config_training
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from util import file_methods, url_methods
from inout.hdf5datasetwriter import HDF5DatasetWriter
from datasets.dataset_loader import DatasetLoader
from augmenting.image_mesh_augmenting import ImageMeshAugmenting

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-bs", "--batch", type=int, default=200,
    help="batch size")
ap.add_argument("-buf", "--bufsize", type=int, default=1000,
    help="buffer size")
ap.add_argument("-zip", "--zipDir", type=str, 
    default=config_training.DATA_PATH,
    help="path to download zip dataset")
ap.add_argument("-et", "--extractdir", type=str,
    default=r"D:\Study\CaoHoc\LuanVan\dataset",
    help="path to extract dataset")
ap.add_argument("-hd", "--hdf5dir", type=str,
    default=r"D:\Study\CaoHoc\LUANVAN\hdf5",
    help="path to output hdf5 dataset")
ap.add_argument("-bm", "--buildmode", type=str,
    default=r"val",
    help="train or val")
ap.add_argument("-ex", "--extenddata", type=bool,
    default=False,
    help="is getting extend data")
ap.add_argument("-dm", "--datamode", type=int, default=5,
    help="small size or normal dataset")
args = vars(ap.parse_args())

DATA_MODE = args["datamode"]
BATCH_SIZE = args["batch"]
BUF_SIZE = args["bufsize"]
BUILD_MODE = args["buildmode"]
EXTEND_DATA = args["extenddata"]
DATA_ZIP_PATH = args["zipDir"]
DATASET_EXTRACT_PATH = args["extractdir"]
OUTPUT_DIR = args["hdf5dir"]
if os.path.exists(DATASET_EXTRACT_PATH) is False:
    os.mkdir(DATASET_EXTRACT_PATH)
if os.path.exists(OUTPUT_DIR) is False:
    os.mkdir(OUTPUT_DIR)

datasetModes = config_training.DATASET_MODES[:]
if EXTEND_DATA is True:
    modeIndexes = [i for i in range(len(datasetModes)) if BUILD_MODE in datasetModes[i]]
else:
    modeIndexes = [i for i in range(len(datasetModes)) if datasetModes[i]==BUILD_MODE]
datasetURLs = [config_training.DATASET_URLS[i] for i in modeIndexes]
datasetZipNames = [config_training.DATASET_ZIP_NAMES[i] for i in modeIndexes]
datasetDirs = [config_training.DATASET_DIRS[i] for i in modeIndexes]

extractedFilePaths = []
for i, (url, datasetDir, zipName) in enumerate(zip(datasetURLs, datasetDirs, datasetZipNames)):
    extractedFilePath = file_methods.getExtractedPath(DATA_ZIP_PATH, zipName, url,
        DATASET_EXTRACT_PATH, datasetDir)
    if extractedFilePath is None:
        print("[ERROR] Can't get data for the dataset {}".format(datasetDir))
        continue
    extractedFilePaths.append(extractedFilePath)

if len(extractedFilePaths) == 0:
    print("[ERROR] No dataset for building")
    exit()

print("[INFO] Loading image and mat paths...")
imageWithMatPaths = []
for extractedFilePath in extractedFilePaths:
    imageWithMatPaths.extend(file_methods.getImageWithMatList(extractedFilePath))
(imagePaths, matPaths) = zip(*imageWithMatPaths)

dl = DatasetLoader()
# construct a list pairing the training, validation
# image paths along with their corresponding labels and output HDF5
# files
HDF5_PATH = path.sep.join([OUTPUT_DIR, BUILD_MODE+".hdf5"])

# loop over the dataset 
rawImageMaxSize = config.RAW_IMAGE_SIZE
rawMatMaxSize = config.RAW_MAT_SIZE
(uvwidth, uvheight, uvdepth) = (config.UV_WIDTH, config.UV_HEIGHT, config.COLOR_CHANNEL)
DATA_TYPE_MODE = config.DATA_TYPE_MODES[DATA_MODE]
imageDType = DATA_TYPE_MODE[0]
uvmapDType = DATA_TYPE_MODE[1]

print("[INFO] Loading and writing data to HDF5 dataset for {}...".format(HDF5_PATH))

# create HDF5 writer
total = len(imagePaths)
imageDims = (total, rawImageMaxSize)
saveUVMapSize = False
if (uvmapDType is "byte"):
    saveUVMapSize = True
    uvmapShape = (rawMatMaxSize,)
    uvmapDims = (total, *uvmapShape)
else:
    uvmapShape = (uvwidth, uvheight, uvdepth)
    uvmapDims = (total, *uvmapShape)
writer = HDF5DatasetWriter(HDF5_PATH, imageDims, uvmapDims,
    imageDType=imageDType, uvmapDType=uvmapDType, saveUVMapSize=saveUVMapSize, bufSize=BUF_SIZE)
bs = BATCH_SIZE

print("[INFO] Processing for total {} images...".format(len(imagePaths)))
pbar = tqdm(desc="Generating data", total=len(imagePaths))  
# loop over the images in batches    
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of paths of image and mask
    batchImagePaths = imagePaths[i:i + bs]
    batchMatPaths = matPaths[i:i + bs]

    (images, uvmaps) = dl.load(batchImagePaths, batchMatPaths, None, DATA_MODE, False)

    # extract the batch of images and labels
    output_batch_images = []
    output_batch_uvmaps = []
    output_batch_uvmap_sizes = []

    # loop over the images and mask in the current batch
    for (image, uvmap) in zip(images, uvmaps):
        # load the input image and preprocess the image by expanding the dimensions
        pad_width = (0, rawImageMaxSize - image.shape[0])
        output_image = np.expand_dims(np.pad(image, pad_width), axis=0)

        if (uvmapDType is "byte"):
            pad_width = (0, rawMatMaxSize - uvmap.shape[0])
            output_uvmap = np.expand_dims(np.pad(uvmap, pad_width), axis=0)
            output_uvmap_size = np.expand_dims(uvmap.shape[0], axis=0)
            output_batch_uvmap_sizes.append(output_uvmap_size)
        else:
            output_uvmap = np.expand_dims(uvmap, axis=0)
            output_uvmap_size = None

        # add the image and mask to the batch
        output_batch_images.append(output_image)
        output_batch_uvmaps.append(output_uvmap)

    # reshape the features
    output_batch_images = np.vstack(output_batch_images)
    output_batch_uvmaps = np.vstack(output_batch_uvmaps)
    images = output_batch_images.reshape((output_batch_images.shape[0], rawImageMaxSize))
    uvmaps = output_batch_uvmaps.reshape((output_batch_uvmaps.shape[0], *uvmapShape))
    uvmapSizes = None
    if (len(output_batch_uvmap_sizes) > 0):
        output_batch_uvmap_sizes = np.vstack(output_batch_uvmap_sizes)
        uvmapSizes = output_batch_uvmap_sizes.reshape((output_batch_uvmap_sizes.shape[0],))

    # add the features and masks to our HDF5 dataset
    writer.add(images, uvmaps, uvmapSizes)

    pbar.update(len(batchImagePaths))
pbar.close()
# close the HDF5 writer
writer.close()