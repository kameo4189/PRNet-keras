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

class FaceAlignmentSparseEvaluating:
    i3e = Image3DExtracting()
    bd = BoundingboxDetecting()
    ie = ImageGenerator()
    ie.logging = False
    uvr = UVMapRestoring()
    kptPosHeaders = [header 
        for headers in [["kpt_{:02d}_X".format(i), "kpt_{:02d}_Y".format(i), "kpt_{:02d}_Z".format(i)] 
        for i in range(1, 69)] for header in headers]
    groundTruthHeaders = ["path"] + kptPosHeaders + ["bbSize"]
    predictHeaders = ["path"] + kptPosHeaders
    errorHeaders = ["path", "error"]
    errorDistributionHeaders = ["errorLimit", "ratio"]
    csvPathIndex = 0
    csvKptPosCount = 68 * 3
    csvKptStartIndex = csvPathIndex + 1
    csvKptEndIndex = csvKptStartIndex + csvKptPosCount - 1
    csvBbSizeIndex = csvKptEndIndex + 1

    lineStyle = "b-"
    mode = None
    evaluateMode = "sparse"

    def evaluate(self, imagePaths, matPaths, batchSize,
        modelPath, groundTruthCsv, predictCsv, errorCsv, errorDistributionCsv,
        errorDistributionFig, mode):
        self.mode = mode
        if (os.path.exists(groundTruthCsv) is False):
            self.__writeGroundTruthKptPos(imagePaths, matPaths, batchSize, groundTruthCsv)
        if (os.path.exists(predictCsv) is False):
            self.__writePredictKptPos(imagePaths, matPaths, batchSize, modelPath, predictCsv)
        if (os.path.exists(errorCsv) is False):
            self.__writingNormalizeMeanError(batchSize, groundTruthCsv, predictCsv, errorCsv, mode)
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

    def __writingNormalizeMeanError(self, batchSize, groundTruthCsv, predictCsv, errorCsv, mode):
        print("Writing normalize mean error of keypoints to {}...".format(errorCsv))
        f = open(errorCsv, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(self.errorHeaders)

        groundTruthFile = open(groundTruthCsv, "r", newline='')
        predictFile = open(predictCsv, "r", newline='')
        groundTruthReader = csv.reader(groundTruthFile)
        predictReader = csv.reader(predictFile)
        next(groundTruthReader)
        next(predictReader)

        pbar = tqdm(desc="Writing")
        batchImagePath = []
        batchGroundTruthKptPos = []
        batchPredictKptPos = []
        batchBBSize = []
        for (groundTruthRow, predictRow) in zip(groundTruthReader, predictReader):
            imagePath = groundTruthRow[self.csvPathIndex]
            groundTruthKptPos = np.array(groundTruthRow[self.csvKptStartIndex:self.csvKptEndIndex+1]).astype("float32").reshape((68, 3))
            groundTruthBBSze = np.array(groundTruthRow[self.csvBbSizeIndex]).astype("float32")
            predictKptPos = np.array(predictRow[self.csvKptStartIndex:self.csvKptEndIndex+1]).astype("float32").reshape((68, 3))
            
            batchImagePath.append(imagePath)
            batchGroundTruthKptPos.append(groundTruthKptPos)
            batchPredictKptPos.append(predictKptPos)
            batchBBSize.append(groundTruthBBSze)

            if (len(batchImagePath) >= batchSize):
                batchNormalizeMeanErrors = self.__calculateNormalizeError(batchGroundTruthKptPos, 
                    batchPredictKptPos, batchBBSize, mode)
                writer.writerows(zip(batchImagePath, batchNormalizeMeanErrors))

                batchImagePath = []
                batchGroundTruthKptPos = []
                batchPredictKptPos = []
                batchBBSize = []

                pbar.update(len(batchImagePath))
        
        if (len(batchImagePath) > 0):
            batchNormalizeMeanErrors = self.__calculateNormalizeError(batchGroundTruthKptPos,
                batchPredictKptPos, batchBBSize, mode)
            writer.writerows(zip(batchImagePath, batchNormalizeMeanErrors))

            pbar.update(len(batchImagePath))

        f.close()
        groundTruthFile.close()
        predictFile.close()
        pbar.close()
    
    def __calculateNormalizeError(self, groundTruthKptPoses, predictKptPoses, bbSizes, mode):
        groundTruthKptPoses = np.array(groundTruthKptPoses)
        predictKptPoses = np.array(predictKptPoses)
        bbSizes = np.array(bbSizes)

        distances = np.power((groundTruthKptPoses - predictKptPoses), 2)
        if mode is "2D":
            distances = np.sqrt(np.sum(distances[:,:,:2], axis=2))
        else:
            distances = np.sqrt(np.sum(distances, axis=2))
        errors = distances / bbSizes[:,np.newaxis]
        errors = np.mean(errors, axis=1)

        return errors

    def __writeGroundTruthKptPos(self, imagePaths, matPaths, batchSize, outputPath):
        print("Writing position of ground truth keypoints to {}...".format(outputPath))
        f = open(outputPath, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(self.groundTruthHeaders)

        pbar = tqdm(desc="Writing", total=len(imagePaths)) 
        for i in range(0, len(imagePaths), batchSize):
            images = [file_methods.readImage(path) for path in imagePaths[i:i+batchSize]]
            mats = [file_methods.readMat(path) for path in matPaths[i:i+batchSize]]

            meshInfos = [self.i3e.preprocess(mat) for mat in mats]
            for i in range(len(meshInfos)):
                meshInfos[i].vertices[:, 2] = meshInfos[i].vertices[:, 2] - np.min(meshInfos[i].vertices[:, 2])
            kptPoses = [meshInfo.kptPos.astype("float32") for meshInfo in meshInfos]

            kptBBs = [self.bd.detect(point, image.shape[0], image.shape[1], 
                False, normalBB=True) for (image, point) in zip(images, kptPoses)]
            bbSizes = [np.sqrt(np.power(r-l,2) + np.power(b-t,2)).astype("float32") 
                for (l,r,t,b) in kptBBs]

            kptPoses = [kptPos.reshape(-1).tolist() for kptPos in kptPoses]
            writer.writerows([(items[0], *items[1], items[2]) for items in zip(imagePaths, kptPoses, bbSizes)])

            pbar.update(len(images))

        f.close()
        pbar.close()

    def __writePredictKptPos(self, imagePaths, matPaths, batchSize, modelPath, outputPath):
        print("Writing position of predicting keypoints to {}...".format(outputPath))
        f = open(outputPath, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(self.predictHeaders)
        from tensorflow.keras.models import load_model
        model = load_model(modelPath, compile=False)

        pbar = tqdm(desc="Writing", total=len(imagePaths))
        for i in range(0, len(imagePaths), batchSize):
            images = [file_methods.readImage(path) for path in imagePaths[i:i+batchSize]]
            mats = [file_methods.readMat(path) for path in matPaths[i:i+batchSize]]

            faceImageInfos = self.ie.generateMulti(images, mats)
            faceImageInfos = [faceImageInfo[0] for faceImageInfo in faceImageInfos]
            (_, faceImages, tforms) = zip(*faceImageInfos)
            uvmaps = model.predict(np.array(faceImages))
            uvmaps = [uvmap for uvmap in uvmaps]
            restoreMeshInfos = [self.uvr.postprocess(image, uvmap, tform) for (image, uvmap, tform) in zip(images, uvmaps, tforms)]
            predictKptPoses = [mesh.kptPos.astype("float32") for mesh in restoreMeshInfos]

            predictKptPoses = [kptPos.reshape(-1).tolist() for kptPos in predictKptPoses]
            writer.writerows([(items[0], *items[1]) for items in zip(imagePaths, predictKptPoses)])

            pbar.update(len(images))

        f.close()
        pbar.close()

