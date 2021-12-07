import os
import sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
os.chdir(__location__)

import argparse
import matplotlib.pyplot as plt
from util import file_methods, mesh_render, mesh_display
from util.resful_api_methods import OutputData, InputData, ImageMatData, ImageMatEvalData, InputEvalData, OutputEvalData
from preprocessing.image3D_extracting import Image3DExtracting
import requests

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
    # default = r"D:\Pictures\test\IMG_20200704_093646.jpg",
    default = r"D:\Pictures\test\image00060.jpg",
    help="path to image")
ap.add_argument("-d", "--imagedir", type=str, required=False,
    default = r"D:\Pictures\test",
    #default = None,
    help="path to image")
ap.add_argument("-u", "--serverurl", type=str,
    #default = r"https://cdba-34-141-153-145.ngrok.io",
    default = r"http://localhost:10000",
    help="server url")
args = vars(ap.parse_args())

SERVER_URL = args["serverurl"]
PREDICT_URL = SERVER_URL+ "/predict"
inputData = None
inputMat = None
if ("imagedir" in args and args["imagedir"] is not None):
    imageMatPaths = file_methods.getImageWithMatList(args["imagedir"])
    (imagePaths, matPaths) = zip(*imageMatPaths)
else:
    imagePaths = [args["image"]]
    matPaths = [file_methods.getFilePathWithOtherExt(args["image"], ".mat")]

print("[INFO] Reading raw image and mat...")
image2Ds = [file_methods.readImage(path, True) for path in imagePaths]
images = [file_methods.readRawFile(path, True) for path in imagePaths]
mats = [file_methods.readRawFile(path, True) for path in matPaths]
rawImages = [file_methods.bytesToString(image) for image in images]
rawMatsList = [[file_methods.bytesToString(mat)] for mat in mats]

print("[INFO] Creating request data...")
imageMatDataList = [ImageMatData(rawImage=rawImage, rawMats=rawMats) for 
    (rawImage, rawMats) in zip(rawImages, rawMatsList)]
requestData = InputData(imageMatList=imageMatDataList)

print("[INFO] Sending request and waiting for response...")
response = requests.post(PREDICT_URL, json=requestData.dict())
if response.status_code != 200:
    print("[ERROR] Server return error result: {}".format(response.content))
    exit()

print("[INFO] Creating mesh info from response...")
i3e = Image3DExtracting()
outputData = OutputData(**response.json())
meshInfosList = [[] for _ in range(len(images))]
for i, imageMeshData in enumerate(outputData.imageMeshList):
    for faceMeshData in imageMeshData.faceMeshList:
        rawMesh = file_methods.stringToBytes(faceMeshData.rawMesh)
        rawMesh = file_methods.readMat(rawMesh)
        meshInfo = i3e.preprocess(rawMesh)
        faceBB = ((faceMeshData.faceBB[0], faceMeshData.faceBB[1]), faceMeshData.faceBB[2])
        meshInfosList[i].append((meshInfo, faceBB))

print("[INFO] Sending predicted mesh and waiting for evaluation...")
rawOrgMatsList = rawMatsList
rawPredictMatsList = [[] for _ in range(len(outputData.imageMeshList))]
for i, imageMeshData in enumerate(outputData.imageMeshList):
    for faceMeshData in imageMeshData.faceMeshList:
        rawPredictMatsList[i].append(faceMeshData.rawMesh)
imageMatDataList = [ImageMatEvalData(rawImage=rawImage, rawOrgMats=rawOrgMats, rawPredictMats=rawPredictMats) for 
    (rawImage, rawOrgMats, rawPredictMats) in zip(rawImages, rawOrgMatsList, rawPredictMatsList)]
requestData = InputEvalData(imageMatList=imageMatDataList)

EVAL_URL = SERVER_URL+ "/evaluate"
response = requests.post(EVAL_URL, json=requestData.dict())
if response.status_code != 200:
    print("[ERROR] Server return error result: {}".format(response.content))
    exit()
outputEvalData = OutputEvalData(**response.json())
print(outputEvalData)
exit()

print("[INFO] Displaying mesh info result...")
from preprocessing.image3D_to_2D import Image3DTo2D
i3t2 = Image3DTo2D()
for i, (image, meshInfos) in enumerate(zip(image2Ds, meshInfosList)):
    plt.subplot(1, len(image2Ds), i+1)
    (h,w,_) = image.shape
    poseBoxImage = image.copy()
    for (meshInfo, faceBB) in meshInfos:
        poseBoxImage = mesh_render.plot_pose_box(poseBoxImage, meshInfo)
        meshFace = i3t2.preprocess(meshInfo, h, w, False)
        meshFaceKpts = i3t2.drawKeypoints(meshFace, meshInfo)
        mesh_display.displayPointCloudColor(meshInfo)
    plt.imshow(poseBoxImage)
    plt.imshow(meshFaceKpts, alpha=.5)
plt.show(block=True)
