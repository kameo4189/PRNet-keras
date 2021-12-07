import os
import csv

from configure import config_evaluating
from util import file_methods
from preprocessing.image3D_extracting import Image3DExtracting
from generating.image_generator import ImageGenerator
from detecting.boundingbox_detecting import BoundingboxDetecting
from postprocessing.uvmap_restoring import UVMapRestoring
import numpy as np
import csv
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

class FaceAlignmentDenseEvaluating:
    i3e = Image3DExtracting()
    bd = BoundingboxDetecting()
    ie = ImageGenerator()
    uvr = UVMapRestoring()
    ie.logging = False

    errorHeaders = ["path", "error"]
    errorDistributionHeaders = ["errorLimit", "ratio"]

    lineStyle = "g-"
    mode = None
    evaluateMode = "sparse"

    def evaluate(self, imagePaths, matPaths, batchSize, modelPath, errorCsv, errorDistributionCsv,
        errorDistributionFig, mode):
        self.mode = mode
        if (os.path.exists(errorCsv) is False):
            self.__writingNormalizeMeanError(imagePaths, matPaths, batchSize, modelPath, errorCsv, mode)
        if (os.path.exists(errorDistributionCsv) is False):
            self.__writingCumulativeErrorsDistribution(errorCsv, errorDistributionCsv)
        self.__drawingCumulativeErrorsDistributionFig(errorDistributionCsv, errorDistributionFig)

    def __drawingCumulativeErrorsDistributionFig(self, errorDistributionCsv, errorDistributionFig):
        print("Drawing cumulative errors distribution of keypoints to {}...".format(errorDistributionFig))
        errorDistributionFile = open(errorDistributionCsv, "r", newline='')
        reader = csv.reader(errorDistributionFile)
        errorDistributions = [row for row in reader]
        errorDistributionFile.close()
        errorDistributions = errorDistributions[1:]
        (errorLimits, errorRatios, errorMeans) = zip(*errorDistributions)
        errorLimits = np.array(errorLimits).astype("float32")
        errorRatios = np.array(errorRatios).astype("float32")
        errorMeanRatio = float(errorMeans[-1]) * 100

        plt.clf()
        plt.xlim(0,7)
        plt.ylim(0,100)
        plt.yticks(np.arange(0,110,10))
        plt.xticks(np.arange(0,11,1))
        plt.grid()
        plt.title('NME (%)', fontsize=20)
        plt.xlabel('NME (%)', fontsize=16)
        plt.ylabel('Test Images (%)', fontsize=16)
        plt.plot(errorLimits*100, errorRatios*100, self.lineStyle,
            label='PRN ({} {}): {:.2f}'.format(self.mode, self.evaluateMode, errorMeanRatio), 
            lw=3)
        plt.legend(loc=4, fontsize=16)
        plt.savefig(errorDistributionFig)
        # plt.show()

    def __writingCumulativeErrorsDistribution(self, errorCsv, errorDistributionCsv):
        print("Writing cumulative errors distribution of keypoints to {}...".format(errorDistributionCsv))
        
        errorFile = open(errorCsv, "r", newline='')
        reader = csv.reader(errorFile)
        errors = [row for row in reader]
        errorFile.close()
        errors = errors[1:]
        (_, errors) = zip(*errors)
        errors = np.array(errors).astype("float32")

        errorDistributionFile = open(errorDistributionCsv, "w", newline='')
        writer = csv.writer(errorDistributionFile)
        writer.writerow(self.errorDistributionHeaders)        

        errorLimits = np.linspace(0,1,1000)
        errorRatios = [(errors < errorLimits[i]).sum()/float(len(errors)) for i in range(1000)]
        errorMeans = [errors[errors < errorLimits[i]].mean() for i in range(1000)]

        writer.writerows(zip(errorLimits, errorRatios, errorMeans))
        errorDistributionFile.close()        

    def __writingNormalizeMeanError(self, imagePaths, matPaths, batchSize, modelPath, errorCsv, mode):
        print("Writing normalize mean error of keypoints to {}...".format(errorCsv))
        from tensorflow.keras.models import load_model
        model = load_model(modelPath, compile=False)

        f = open(errorCsv, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(self.errorHeaders)

        pbar = tqdm(desc="Writing", total=len(imagePaths))
        for i in range(0, len(imagePaths), batchSize):
            batchImagePath = imagePaths[i:i+batchSize]
            batchMatPath = matPaths[i:i+batchSize]
            images = [file_methods.readImage(path) for path in batchImagePath]
            mats = [file_methods.readMat(path) for path in batchMatPath]

            # get ground truth meshs
            groundTruthMeshs = self.__getGroundTruthMesh(mats)
            
            # get predicting meshs
            restoreMeshs = self.__getPredictMesh(images, mats, model)

            # get bounding box sizes
            bbs = [self.bd.detect(mesh.vertices, image.shape[0], image.shape[1], 
                False, normalBB=True) for (image, mesh) in zip(images, groundTruthMeshs)]
            bbSizes = [np.sqrt(np.power(r-l,2) + np.power(b-t,2)).astype("float32") 
                for (l,r,t,b) in bbs]

            # get mesh info pairs for calculating
            (groundTruthMeshs, restoreMeshs) = self.__getMeshPair(groundTruthMeshs, restoreMeshs)

            # calculate nme
            normalizeMeanErrors = self.__calculateNormalizeError(groundTruthMeshs, 
                    restoreMeshs, bbSizes, mode)
            writer.writerows(zip(batchImagePath, normalizeMeanErrors))

            pbar.update(len(batchImagePath))

        f.close()
        pbar.close()

    def __getGroundTruthMesh(self, mats):
        meshs = [self.i3e.preprocess(mat) for mat in mats]
        for i in range(len(meshs)):
            meshs[i].vertices[:, 2] = meshs[i].vertices[:, 2] - np.min(meshs[i].vertices[:, 2])
        return meshs
    
    def __getPredictMesh(self, images, mats, model):
        faceImageInfos = self.ie.generateMulti(images, mats)
        faceImageInfos = [faceImageInfo[0] for faceImageInfo in faceImageInfos]
        (_, faceImages, tforms) = zip(*faceImageInfos)
        uvmaps = model.predict(np.array(faceImages))
        uvmaps = [uvmap for uvmap in uvmaps]
        meshs = [self.uvr.postprocess(image, uvmap, tform) for (image, uvmap, tform) in zip(images, uvmaps, tforms)]
        return meshs

    def __getMeshPair(self, groundTruthMeshs, restoreMeshs):
        gtMeshs = [mesh.toCommonFaceRegionMesh() for mesh in groundTruthMeshs]
        return (gtMeshs, restoreMeshs)

    def __calculateNormalizeError(self, groundTruthMeshs, restoreMeshs, bbSizes, mode):
        groundTruthPoses = [mesh.vertices for mesh in groundTruthMeshs]
        predictPoses = [mesh.vertices for mesh in restoreMeshs]
        groundTruthPoses = np.array(groundTruthPoses)
        predictPoses = np.array(predictPoses)
        bbSizes = np.array(bbSizes)

        distances = np.power((groundTruthPoses - predictPoses), 2)
        if mode is "2D":
            distances = np.sqrt(np.sum(distances[:,:,:2], axis=2))
        else:
            distances = np.sqrt(np.sum(distances, axis=2))
        errors = distances / bbSizes[:,np.newaxis]
        errors = np.mean(errors, axis=1)

        return errors

