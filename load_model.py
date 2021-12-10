import os
import sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
os.chdir(__location__)

from configure import config_training as confa
from configure import config
from tensorflow.keras.models import load_model
from skimage import io
import numpy as np
import argparse
import matplotlib.pyplot as plt
from generating.image_generator import ImageGenerator
from util import file_methods

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
    # default = r"D:\Pictures\test\FB_IMG_1612923425427.jpg",
    default = r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000\image00010.jpg",
    #default = r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000_augment\image02536_0001.jpg",
	#default = r"K:\Study\CaoHoc\LuanVan\dataset\300W_LP\LFPW\LFPW_image_test_0003_0.jpg",
    help="path to image")
ap.add_argument("-m", "--model", type=str,
    default=r"M:\My Drive\CaoHoc\LUANVAN\SourceCode\output_wtmse_0.0001\best_eval_loss.model",
    help="path to model")
args = vars(ap.parse_args())

IMAGE_PATH = args["image"]
MODEL_PATH = args["model"]

print("[INFO] Reading image...")
image = io.imread(IMAGE_PATH)

print("[INFO] Creating input face image...")
ig = ImageGenerator()
matPath = file_methods.getFilePathWithOtherExt(IMAGE_PATH, ".mat")

faceInfoList = ig.generate(image, matPath)
if (len(faceInfoList) == 0):
    print("[WARNING] No face detected")
    exit()
faceBBs, faceImages, tforms = zip(*faceInfoList)

print("[INFO] Loading pre-trained network...")
model = load_model(MODEL_PATH, compile=False)

print("[INFO] Predicting for uv position map...")
uvmaps = model.predict(np.array(faceImages))
uvmaps = [uvmap for uvmap in uvmaps]
plt.figure(1)
for i, (faceImage, uvmap) in enumerate(zip(faceImages, uvmaps)):
    stackImage = np.concatenate((faceImage, uvmap), axis=1)
    plt.subplot(len(faceInfoList), 1, i+1)
    plt.imshow(stackImage)

from postprocessing.uvmap_restoring import UVMapRestoring
from postprocessing.obj_creating import ObjCreating
from preprocessing.image3D_to_2D import Image3DTo2D
from util import mesh_display
from util import mesh_render
uvr = UVMapRestoring()
oc = ObjCreating()
i3t2 = Image3DTo2D()
(h, w) = image.shape[:2]

plt.figure(2)
plt.imshow(image)
for i, (tform, uvmap) in enumerate(zip(tforms, uvmaps)):
    restoredMeshInfo = uvr.postprocess(image, uvmap, tform)
    restoreMeshFace = i3t2.preprocess(restoredMeshInfo, h, w, False)
    restoreMeshFaceKpts = i3t2.drawKeypoints(restoreMeshFace, restoredMeshInfo)

    mesh_display.displayPointCloudColor(restoredMeshInfo)
    mesh_display.displayMesh(restoredMeshInfo)

    plt.imshow(restoreMeshFaceKpts, alpha=.5)
plt.show()

    # poseBoxImage = mesh_render.plot_pose_box(image, restoredMeshInfo)
    # plt.imshow(poseBoxImage)
    # plt.show()