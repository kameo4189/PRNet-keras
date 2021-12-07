import os
import sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
os.chdir(__location__)

import csv
import json
import traceback
import argparse
import numpy as np
from configure import config_training
from callbacks.trainingmonitor import TrainingMonitor
from callbacks.trainingstatusmonitor import TrainingStatusMonitor
from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.customlrscheduler import CustomLRScheduler
from callbacks.trainingoutputzipping import TrainingOutputZipping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from model.nn.conv.position_map_regression import PRN
from model.nn.conv.position_map_regression import loss_funtions
from generating.image_uvmap_sequence_generator import ImageUVMapSequenceGenerator
from augmenting.image_mesh_augmenting import ImageMeshAugmenting

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100,
    help="epochs num")
ap.add_argument("-bs", "--batchsize", type=int, default=16,
    help="batch size")
ap.add_argument("-m", "--model", type=str,
    help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type=int,
    default=0,
    help="epoch to restart training at")
ap.add_argument("-lr", "--learningrate", type=float, default=0.0001,
    help="initial learning rate")
ap.add_argument("-loss", "--lossfunct", type=str,
    default="wtmse",
    help="wtmse or wtrmse")
ap.add_argument("-zipo", "--zipOut", type=str,
    default=None,
    help="path to output zip file of model output files")
ap.add_argument("-dr", "--dropfactor", type=float, default=0.5,
    help="drop factor")
ap.add_argument("-de", "--dropevery", type=int, default=5,
    help="drop every")
ap.add_argument("-hd", "--hdf5dir", type=str,
    default=r"D:\Study\CaoHoc\LUANVAN\HDF5",
    help="path to hdf5 dataset")
ap.add_argument("-at", "--autocontinue", type=bool,
    default=True,
    help="auto continue previous epoch")
ap.add_argument("-q", "--max_queue_size", type=int,
    default=20,
    help="max_queue_size for fit model")
ap.add_argument("-w", "--workers", type=int,
    default=4,
    help="num workers for fit model")
ap.add_argument("-pl", "--preload", type=int,
    default=2,
    help="preload images and mats to memory")
ap.add_argument("-sh", "--shuffle", type=int,
    default=1,
    help="shuffle data at start epoch")
args = vars(ap.parse_args())

# define the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]
DROP_EVERY = args["dropevery"]
DROP_FACTOR = args["dropfactor"]
SHUFFLE = False if args["shuffle"] == 0 else True
INIT_LR = args["learningrate"]
ZIP_OUTPUT_PATH = args["zipOut"]
AUTO_CONTINUE = args["autocontinue"]
MAX_QUEUE_SIZE = args["max_queue_size"]
PRELOAD = args["preload"]
WORKERS = args["workers"]
RANDOM_SEED = 1189
LOSS_FUNCTION_NAME = loss_funtions[args["lossfunct"]][0]
LOSS_FUNCTION = loss_funtions[args["lossfunct"]][1]

OUTPUT_POSFIX = "_".join([args["lossfunct"], str(INIT_LR)])
OUTPUT_DIR = "_".join([config_training.BASE_OUTPUT_PATH, OUTPUT_POSFIX])
config_training.createOutputPaths(OUTPUT_DIR)

# create a class for output text to both screen and file
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
        self.te = open(config_training.TXT_PATH_LOGGING,'a')  # File where you need to keep the logs

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.te.write(data)    # Write the data of stdout here to a text file as well

    def writelines(self, lines):
        self.stream.writelines(lines)
        self.stream.flush()
        self.te.writelines(lines)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.stream.flush()
sys.stdout = Unbuffered(sys.stdout)

if (AUTO_CONTINUE):
    if (os.path.exists(config_training.TRAINING_INFO_PATH)):
        savedInfo = json.loads(open(config_training.TRAINING_INFO_PATH).read())
        differentItems = { k: args[k] for k in args 
                            if k in args and 
                                k not in ["model", "start_epoch", "preload"] and
                                args[k] != savedInfo[k] }
        if (len(differentItems.keys()) > 0):
            print("[ERROR] Training info is different from previous, re-check and run again")
            print(differentItems)
            exit()

# getting previous model and epoch
if (AUTO_CONTINUE):
    if (os.path.exists(config_training.JSON_PATH_TRAINING_STATUS)):
        print("[INFO] Getting last model and epoch number ...")
        trainingStatus = json.loads(open(config_training.JSON_PATH_TRAINING_STATUS).read())
        currentEpoch = trainingStatus["current_epoch"]
        args["model"] = os.sep.join([config_training.CHECKPOINTS_PATH, "epoch_{:04d}.hdf5".format(currentEpoch)])
        if (os.path.exists(args["model"]) is False):
            print("[ERROR] Last model path at {} not exist, re-check and run again".format(args["model"]))
            exit()
        args["start_epoch"] = currentEpoch
        print("Last model {} at epoch {}...".format(args["model"], args["start_epoch"]))
    else:
        print("[INFO] No checkpoint info for continue, training from beginning...")

# save training info
with open(config_training.TRAINING_INFO_PATH, "w") as f:
    f.write(json.dumps(args))

def custom_lr_scheduler(epoch):
    # compute the learning rate for the current epoch
    exp = np.floor(epoch / DROP_EVERY)
    alpha = INIT_LR * (DROP_FACTOR ** exp)

    # return the learning rate
    return float(alpha)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
INIT_EPOCH = args["start_epoch"]
LOAD_MODEL = args["model"]
if LOAD_MODEL is None:
    print("[INFO] Building the model...")
    model = PRN.build(256, 256, 3)
    
    print("Model summary")
    print(model.summary(line_length=150))

    print("Writing model structure to json file...")
    with open(config_training.MODEL_STRUCTURE_PATH, 'w') as f:
        f.write(model.to_json())

# otherwise, load the checkpoint from disk
else:
    print("[INFO] Loading model {}...".format(LOAD_MODEL))
    custom_objects = {LOSS_FUNCTION_NAME: LOSS_FUNCTION}
    model = load_model(LOAD_MODEL, custom_objects=custom_objects)

    # update the learning rate
    print("Old learning rate: {}".format(K.get_value(model.optimizer.lr)))

opt = Adam(learning_rate=INIT_LR)
metrics=["accuracy"]
model.compile(loss=LOSS_FUNCTION, optimizer=opt, metrics=metrics)

print("Initial learning rate: {}".format(K.get_value(model.optimizer.lr)))
print("Current learning rate: {}".format(custom_lr_scheduler(INIT_EPOCH)))

plot_model(model, to_file=config_training.MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True)

print("[INFO] Initializing checkpoint configs...")
trainingmonitor = TrainingMonitor(config_training.FIG_PATH,
    jsonPath=config_training.JSON_PATH,
    startAt=INIT_EPOCH)
checkpointbestevalloss = ModelCheckpoint(config_training.MODEL_PATH_BEST_EVAL_LOSS,
    monitor="val_loss",
    save_best_only=True, verbose=1)
checkpointbestloss = ModelCheckpoint(config_training.MODEL_PATH_BEST_LOSS,
    monitor="loss",
    save_best_only=True, verbose=1)
lrPlotting= CustomLRScheduler(schedule=None,
    mode="epoch",
    figPath=config_training.FIG_PATH_LRS,
    jsonPath=config_training.JSON_PATH_LRS,
    startAt=INIT_EPOCH,
    isSetLr=False)
epochcheckpoint = EpochCheckpoint(config_training.CHECKPOINTS_PATH, every=1,
    startAt=INIT_EPOCH, max_to_keep=3)
learningratescheduler = LearningRateScheduler(custom_lr_scheduler)
callbacks = [trainingmonitor,
             checkpointbestevalloss,
             checkpointbestloss,
             epochcheckpoint,
             learningratescheduler,
             lrPlotting]
trainingstatusmonitor = TrainingStatusMonitor(config_training.JSON_PATH_TRAINING_STATUS,
    jsonMetricsPath=config_training.JSON_PATH, modelCheckPointCallbacks=callbacks,
    startAt=INIT_EPOCH)
callbacks += [trainingstatusmonitor]
if (ZIP_OUTPUT_PATH is not None):
    trainingOutputZipping = TrainingOutputZipping(config_training.OUTPUT_PATH, ZIP_OUTPUT_PATH, every=1,
        startAt=INIT_EPOCH, max_to_keep=3)
    callbacks += [trainingOutputZipping]

print("[INFO] Loading data generator...")
HDF5_DIR = args["hdf5dir"]
hdf5TrainPath = os.path.sep.join([HDF5_DIR, config_training.TRAIN_HDF5_NAME])
hdf5ValPath = os.path.sep.join([HDF5_DIR, config_training.VAL_HDF5_NAME])
ima = ImageMeshAugmenting()
preloadTrain = False
preloadVal = False
if PRELOAD == 1:
    preloadTrain = True
elif PRELOAD == 2:
    preloadTrain = True
    preloadVal = True
elif PRELOAD == 3:
    preloadVal = True
trainGenerator = ImageUVMapSequenceGenerator(hdf5TrainPath, BATCH_SIZE, INIT_EPOCH,
    RANDOM_SEED, aug=ima, shuffle=SHUFFLE, preload=preloadTrain, generateMode=1)
valGenerator = ImageUVMapSequenceGenerator(hdf5ValPath, BATCH_SIZE, INIT_EPOCH,
    RANDOM_SEED, aug=None, shuffle=False, mode="val", preload=preloadVal, useImageNum=2000, generateMode=1)

print("[INFO] Network info...")
NUM_TRAIN_IMAGES = trainGenerator.numImages
NUM_EVAL_IMAGES = valGenerator.numImages
TRAIN_RATIO = int(NUM_TRAIN_IMAGES / (NUM_TRAIN_IMAGES + NUM_EVAL_IMAGES) * 100)
VAL_RATIO = 100 - TRAIN_RATIO
STEPS_PER_EPOCH = len(trainGenerator)
STEPS_VAL_PER_EPOCH = len(valGenerator)
print("Train examples: ", NUM_TRAIN_IMAGES)
print("Eval examples: ", NUM_EVAL_IMAGES)
print("Ratio of train/val: {}/{}".format(TRAIN_RATIO, VAL_RATIO))
print("Batch size: ", BATCH_SIZE)
print("Num steps train per epoch: ", STEPS_PER_EPOCH)
print("Num steps val per epoch: ", STEPS_VAL_PER_EPOCH)
print("Initial learning rate: ", INIT_LR)
print("Loss function: ", LOSS_FUNCTION_NAME)

print("[INFO] Training the network...")
# train the network
H = model.fit(x=trainGenerator,
              validation_data=valGenerator,
              epochs=NUM_EPOCHS,
              callbacks=callbacks,
              verbose=1,
              initial_epoch=INIT_EPOCH,
              max_queue_size=MAX_QUEUE_SIZE,
              workers=WORKERS)