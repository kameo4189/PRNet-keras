import numpy as np
from skimage import io
import os

BFM_UV_PATH = r"resource/BFM/BFM_UV.mat"
BFM_PATH = r"resource/BFM/BFM.mat"
MESH_SIZE = 180

UV_HEIGHT = 256
UV_WIDTH = 256
UV_POS_SCALE = 1.1

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
COLOR_CHANNEL = 3

RAW_IMAGE_SIZE = 60000
RAW_MAT_SIZE = 7000

FACE_INDEXES_PATH = r"resource/UV/face_ind.txt"
FACE_TRIANGLES_PATH = r"resource/UV/triangles.txt"
UV_KPT_INDEXES_PATH = r"resource/UV/uv_kpt_ind.txt"
FACE_INDEXES_TO_ORG_TRIANGLES_PATH = r"resource/UV/uv_face_ind_to_org_triangles.txt"
FACE_INDEXES_TO_ORG_TRIWEIGHTS_PATH = r"resource/UV/uv_face_ind_to_org_triweights.txt"

UV_KPT_INDEXES = np.loadtxt(UV_KPT_INDEXES_PATH).astype(np.int32) # 2 x 68 get kpt
FACE_INDEXES = np.loadtxt(FACE_INDEXES_PATH).astype(np.int32) # get valid vertices in the pos map
FACE_TRIANGLES = np.loadtxt(FACE_TRIANGLES_PATH).astype(np.int32) # ntri x 3
FACE_INDEXES_TO_ORG_TRIANGLES = np.loadtxt(FACE_INDEXES_TO_ORG_TRIANGLES_PATH).astype(np.int32)
FACE_INDEXES_TO_ORG_TRIWEIGHTS = np.loadtxt(FACE_INDEXES_TO_ORG_TRIWEIGHTS_PATH).astype(np.float32)

ORG_BB_EXTENDED_RATIO = 0.5
BB_MARGIN_RATIO = 0.1
BB_MARGIN_RANDOM_RANGE = (-1.0, 1.0)
BB_SIZE_RANDOM_RANGE = (0.9, 1.1)

UV_FACE_MASK_PATH = r"resource/MASK/uv_face_mask.png"
UV_KPT_MASK_PATH = r"resource/MASK/uv_kpt_mask.png"
UV_WEIGHT_MASK_PATH = r"resource/MASK/uv_weight_mask.png"
UV_WEIGHT_LOSS_MASK_PATH = r"resource/MASK/uv_weight_loss_mask.png"

CANONICAL_VERTICES_PATH = r"resource/UV/canonical_vertices.npy"

def createWeightedLossMask():
    faceMaskImage = io.imread(UV_FACE_MASK_PATH)
    kptMaskImage = io.imread(UV_KPT_MASK_PATH)
    weightMaskImage = io.imread(UV_WEIGHT_MASK_PATH)

    facePos = np.where(faceMaskImage != 0)
    kptPos1 = np.where(kptMaskImage != 0)
    kptPos2 = np.where(weightMaskImage > 64)
    eyeNoseMouthPos = np.where(weightMaskImage == 64)

    weightMask = np.zeros((UV_WIDTH, UV_HEIGHT), np.uint8)
    weightMask[facePos] = 3
    weightMask[eyeNoseMouthPos] = 4
    weightMask[kptPos1] = 16
    weightMask[kptPos2] = 16

    io.imsave(UV_WEIGHT_LOSS_MASK_PATH, weightMask)

if (os.path.exists(UV_WEIGHT_LOSS_MASK_PATH) is False):
    createWeightedLossMask()
UV_WEIGHT_LOSS_MASK = io.imread(UV_WEIGHT_LOSS_MASK_PATH)

DLIB_DETECTION_MODEL_PATH = r"resource/DLIB/mmod_human_face_detector.dat"

# for saving disk space, image is saved as ubyte and uvmap is saved as float32 instead of float64
DATA_TYPE_MODES = {0: ("uint8", "float32"), 
    1: ("float32", "float32"), 
    2: ("float64", "float64"),
    3: ("float32", "float16"),
    4: ("byte", "float16"),
    5: ("byte", "byte")}
DATA_TYPE_MODE_DEFAULT = 2
