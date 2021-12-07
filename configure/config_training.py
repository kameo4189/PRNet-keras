import os
from os import path

# url to dataset
DATA_PATH = "data"
NPY_POSMAP_DATA_PATH = os.path.sep.join([DATA_PATH, "npy"])
DATASET_URLS = [r"http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip",
                r"https://drive.google.com/u/0/uc?export=download&confirm=RQoB&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k",
                None]
DATASET_ZIP_NAMES = ["AFLW2000-3D.zip", "300W-LP.zip", "AFLW2000_augment.zip"]
DATASET_DIRS = ["AFLW2000", "300W_LP", "AFLW2000_augment"]
DATASET_SPLIT_MODES = ["FILE_NAME", "SUBJECT", "SUBJECT"]
DATASET_MODES = ["val", "train", "val_augment"]

# define the path to the output training, validation, and testing
# HDF5 files
DATA_DIR = r"data"
HDF5_DATASET_DIR = path.sep.join([DATA_DIR, "hdf5"])
if path.exists(DATA_DIR) is False:
    os.mkdir(DATA_DIR)
if path.exists(HDF5_DATASET_DIR) is False:
    os.mkdir(HDF5_DATASET_DIR)
TRAIN_HDF5_FORMAT_NAME = "train{}.hdf5"
TRAIN_HDF5_NAME = "train.hdf5"
VAL_HDF5_FORMAT_NAME = "val{}.hdf5"
VAL_HDF5_NAME = "val.hdf5"
VAL_HDF5_NAMES = ["val{}.hdf5".format("_" + dsName) for dsName in DATASET_DIRS]

# path to train and eval file paths
TRAIN_VAL_SPLIT_PATHS_NAME = "train_test_split.csv"
TRAIN_VAL_SPLIT_PATHS_FORMAT_NAME = "train_test_split{}.csv"

BEST_VAL_LOSS_MODEL_FILE_NAME = "best_eval_loss.model"
BEST_LOSS_MODEL_FILE_NAME = "best_loss.model"

# define the path to the output directory used for storing plots,
# classification reports, etc.
BASE_OUTPUT_PATH = r"output"
def createOutputPaths(baseOutputPath=BASE_OUTPUT_PATH):
    global OUTPUT_PATH, CHECKPOINTS_PATH, MODEL_PLOT_PATH, FIG_PATH, JSON_PATH
    global REPORT_PATH, FIG_PATH_LRS, JSON_PATH_LRS, TXT_PATH_LOGGING, JSON_PATH_TRAINING_STATUS
    global MODEL_PATH_BEST_EVAL_LOSS, MODEL_PATH_BEST_EVAL_ACC, MODEL_PATH_BEST_LOSS, MODEL_PATH_BEST_ACC
    global MODEL_STRUCTURE_PATH, TRAINING_INFO_PATH

    OUTPUT_PATH = baseOutputPath
    CHECKPOINTS_PATH = path.sep.join([OUTPUT_PATH, "model_checkpoints"])
    if path.exists(OUTPUT_PATH) is False:
        os.mkdir(OUTPUT_PATH)
    if path.exists(CHECKPOINTS_PATH) is False:
        os.mkdir(CHECKPOINTS_PATH)
    MODEL_PLOT_PATH = path.sep.join([OUTPUT_PATH, "model_plot.png"])
    FIG_PATH = path.sep.join([OUTPUT_PATH, "metrics.png"])
    JSON_PATH = path.sep.join([OUTPUT_PATH, "metrics.json"])
    REPORT_PATH = path.sep.join([OUTPUT_PATH, "report.txt"])
    FIG_PATH_LRS = path.sep.join([OUTPUT_PATH, "lrs.png"])
    JSON_PATH_LRS = path.sep.join([OUTPUT_PATH, "lrs.json"])
    TXT_PATH_LOGGING = path.sep.join([OUTPUT_PATH, "training_log.txt"])
    JSON_PATH_TRAINING_STATUS = path.sep.join([OUTPUT_PATH, "training_status.json"])

    # path to the output model file
    MODEL_PATH_BEST_EVAL_LOSS = path.sep.join([OUTPUT_PATH, BEST_VAL_LOSS_MODEL_FILE_NAME])
    MODEL_PATH_BEST_EVAL_ACC = path.sep.join([OUTPUT_PATH, "best_eval_acc.model"])
    MODEL_PATH_BEST_LOSS = path.sep.join([OUTPUT_PATH, BEST_LOSS_MODEL_FILE_NAME])
    MODEL_PATH_BEST_ACC = path.sep.join([OUTPUT_PATH, "best_acc.model"])

    # path to the output model structure by json
    MODEL_STRUCTURE_PATH = path.sep.join([OUTPUT_PATH, "structure.json"])

    # path to the output model structure by json
    TRAINING_INFO_PATH = path.sep.join([OUTPUT_PATH, "training_info.json"])

createOutputPaths()