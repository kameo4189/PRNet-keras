import os
import sys
import argparse

from configure import config_training
from configure import config_evaluating
from util import file_methods
from evaluating.face_restoring_evaluating import FaceRestoringEvaluating
from message_logging.server_logging import ServerLogger

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--modeldir", type=str,
    default = r"M:\My Drive\CaoHoc\LUANVAN\SourceCode\output_wtmse_0.0001_WBA_CS",
    help="path to image")
ap.add_argument("-d", "--datadir", type=str,
    default=r"D:\Study\CaoHoc\LUANVAN\Dataset",
    help="path to data dir")
ap.add_argument("-hf", "--hdf5", type=str,
    default=r"D:\Study\CaoHoc\LUANVAN\HDF5\val.hdf5",
    help="path to hdf5 data file")
ap.add_argument("-bs", "--batchsize", type=int,
    default=10,
    help="batch size")
ap.add_argument("-in", "--imagenum", type=int,
    default=-1,
    help="number if image for eval")
ap.add_argument("-o", "--outputdir", type=str,
    default = config_evaluating.BASE_EVALUATION_PATH,
    help="path to output dir")
ap.add_argument("-i", "--evalidxs", nargs='+', type=int,
    default = list(range(6)),
    help="path to output dir")
ap.add_argument("-sl", "--serverlog", type=int,
    default=1,
    help="0: no remote log, >0: remote log")
ap.add_argument("-sh", "--shiftZ", type=int,
    default=1,
    help="0: not shift Z to 0, >0: shiftZ to 0")
args = vars(ap.parse_args())

MODEL_DIR = args["modeldir"]
HDF5_PATH = args["hdf5"]
DATASET_EXTRACT_PATH = args["datadir"]
BATCH_SIZE = args["batchsize"]
VAL_MODE = "val"
IMAGE_NUM = args["imagenum"] if args["imagenum"] > 0 else None
SHIFTZ = True if args["shiftZ"] > 0 else False
OUTPUT_DIR = args["outputdir"]
EVAL_IDXS = args["evalidxs"]
DATA_ZIP_PATH = config_training.DATA_PATH
SERVER_LOG = args["serverlog"]

modelPath = os.path.sep.join([MODEL_DIR, config_training.BEST_VAL_LOSS_MODEL_FILE_NAME])
modelDirName = file_methods.getFileName(MODEL_DIR)

logger = ServerLogger(logSource=modelDirName)
if (SERVER_LOG > 0):
    sys.stdout = logger

evalOutputDir = os.path.sep.join([OUTPUT_DIR, modelDirName])
config_evaluating.createEvaluationInfos(baseEvaluationPath=evalOutputDir)

if HDF5_PATH is not None and os.path.exists(HDF5_PATH):
    imageValPaths = None
    matValPaths = None
elif VAL_MODE not in config_training.DATASET_MODES:
    print("[INFO] Getting image paths from dataset dir {}...".format(DATASET_EXTRACT_PATH))
    imageWithMatPaths = file_methods.getImageWithMatList(DATASET_EXTRACT_PATH)
    (imageValPaths, matValPaths) = zip(*imageWithMatPaths)
else:
    print("[INFO] Getting image paths from {} datasets...".format(VAL_MODE))
    datasetModes = config_training.DATASET_MODES[:]
    modeIndexes = [i for i in range(len(datasetModes)) if VAL_MODE in datasetModes[i]]
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

    imageWithMatPaths = []
    for extractedFilePath in extractedFilePaths:
        imageWithMatPaths.extend(file_methods.getImageWithMatList(extractedFilePath))
    (imageValPaths, matValPaths) = zip(*imageWithMatPaths)

fre = FaceRestoringEvaluating(logger, SHIFTZ)
fre.evaluate(imageValPaths, matValPaths, BATCH_SIZE, modelPath, EVAL_IDXS, HDF5_PATH, IMAGE_NUM)