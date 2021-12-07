import os
import sys
import csv

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from configure import config_evaluating
from util import file_methods
from util import interative_closest_point
from preprocessing.image3D_extracting import Image3DExtracting
from generating.image_generator import ImageGenerator
from detecting.boundingbox_detecting import BoundingboxDetecting
from postprocessing.uvmap_restoring import UVMapRestoring
from abc import ABCMeta, abstractmethod
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import shutil

class EvaluatingProcessor(metaclass=ABCMeta):
    def __init__(self, evaluationType):
        self.evaluationInfo = config_evaluating.EVALUATION_INFOS[evaluationType]
        evaluationTypeParts = evaluationType.split("_")
        self.type = "_".join(evaluationTypeParts[:-1])
        self.mode = evaluationTypeParts[-1]

    @abstractmethod
    def getMeshPairs(self, groundTruthMeshs, restoreMeshs):
        pass

    @abstractmethod
    def getPositionPairs(self, groundTruthMeshs, restoreMeshs):
        pass

    @abstractmethod
    def getNormalizedSizes(self, groundTruthMeshs, images):
        pass

    def initializeCsvFile(self, csvType, openType, writeAppend=False):
        if csvType == "NME":
            if openType == "read":
                self.fileReaderNME = open(self.evaluationInfo.FILEPATH_NME, "r", newline='')
                self.readerNME = csv.reader(self.fileReaderNME)
            else:
                if writeAppend:
                    self.fileWriterNME = open(self.evaluationInfo.FILEPATH_NME, "a", newline='')
                else:
                    self.fileWriterNME = open(self.evaluationInfo.FILEPATH_NME, "w", newline='')
                self.writerNME = csv.writer(self.fileWriterNME)
        if csvType == "CED":
            if openType == "read":
                self.fileReaderCED = open(self.evaluationInfo.FILEPATH_CED, "r", newline='')
                self.readerCED = csv.reader(self.fileReaderCED)
            else:
                self.fileWriterCED = open(self.evaluationInfo.FILEPATH_CED, "w", newline='')
                self.writerCED = csv.writer(self.fileWriterCED)

    def writerow(self, items):
        self.writerNME.writerow(items)
    
    def writerows(self, itemsList):
        self.writerNME.writerows(itemsList)
        
    def flush(self):
        self.fileWriterNME.flush()

    def closeCsvFile(self, csvType, openType):
        if csvType == "NME":
            if openType == "read":
                self.fileReaderNME.close()
            else:
                self.fileWriterNME.close()
        if csvType == "CED":
            if openType == "read":
                self.fileReaderCED.close()
            else:
                self.fileWriterCED.close()

class FaceAlignmentSparseEvaluating(EvaluatingProcessor):
    def __init__(self, evaluationType):
        super().__init__(evaluationType)

    def getMeshPairs(self, groundTruthMeshs, restoreMeshs):
        gtMeshs = [gtMesh.toCommonFaceRegionMesh() if gtMesh is not None and gtMesh.vertices.shape[0] != resMesh.vertices.shape[0]
            else gtMesh for (gtMesh, resMesh) in zip(groundTruthMeshs, restoreMeshs)]
        return (gtMeshs, restoreMeshs)

    def getPositionPairs(self, groundTruthMeshs, restoreMeshs):
        gtKptPoss = [mesh.kptPos.astype("float32") if mesh is not None else None for mesh in groundTruthMeshs]
        reKptPoss = [mesh.kptPos.astype("float32") if mesh is not None else None for mesh in restoreMeshs]
        return (gtKptPoss, reKptPoss)

    def getNormalizedSizes(self, groundTruthMeshs, images):
        bbSizes = [mesh.bbSize(image.shape[0], image.shape[1]) if mesh is not None else None
            for (image, mesh) in zip(images, groundTruthMeshs)]
        return bbSizes

class FaceAlignmentDenseEvaluating(EvaluatingProcessor):
    def __init__(self, evaluationType):
        super().__init__(evaluationType)

    def getMeshPairs(self, groundTruthMeshs, restoreMeshs):
        gtMeshs = [gtMesh.toCommonFaceRegionMesh() 
            if gtMesh is not None and gtMesh.vertices.shape[0] != resMesh.vertices.shape[0]
            else gtMesh
            for (gtMesh, resMesh) in zip(groundTruthMeshs, restoreMeshs)]
        return (gtMeshs, restoreMeshs)

    def getPositionPairs(self, groundTruthMeshs, restoreMeshs):
        gtPoss = [mesh.vertices if mesh is not None else None for mesh in groundTruthMeshs]
        rePoss = [mesh.vertices if mesh is not None else None for mesh in restoreMeshs]
        return (gtPoss, rePoss)

    def getNormalizedSizes(self, groundTruthMeshs, images):
        bbSizes = [mesh.bbSize(image.shape[0], image.shape[1]) if mesh is not None else None
            for (image, mesh) in zip(images, groundTruthMeshs)]            
        return bbSizes

class FaceRecontructonEvaluating(FaceAlignmentDenseEvaluating):
    def __init__(self, evaluationType):
        super().__init__(evaluationType)

    def alignMesh(self, fixMesh, moveMesh):
        X_fix = fixMesh.vertices
        X_move = moveMesh.vertices
        
        X_fix_mean = np.mean(X_fix, axis=0)
        X_move_mean = np.mean(X_move, axis=0)
        translations = X_move_mean - X_fix_mean
        X_move = X_move - translations

        (_, X_move_transformed, nearest_idxs) = interative_closest_point.simpleicp(X_fix, X_move, X_move.shape[0]//2, min_change=0.01)

        tranformedMoveMesh = moveMesh.copy()
        tranformedMoveMesh.vertices = X_move_transformed
        nearestFixMesh = fixMesh.copy()
        nearestFixMesh.vertices = nearestFixMesh.vertices[nearest_idxs]
        nearestFixMesh.colors = nearestFixMesh.colors[nearest_idxs]
        nearestFixMesh.triangles = None
        return tranformedMoveMesh, nearestFixMesh

    def getMeshPairs(self, groundTruthMeshs, restoreMeshs):
        interative_closest_point.previewing = False
        interative_closest_point.logging = False
        meshPairs = [self.alignMesh(gtMesh, restoredMesh) 
            if gtMesh is not None else (None, None)
            for gtMesh, restoredMesh 
            in zip(groundTruthMeshs, restoreMeshs)]
        (alignedMeshs, fixMeshs) = zip(*meshPairs)       
        return (alignedMeshs, fixMeshs)

    def getNormalizedSizes(self, groundTruthMeshs, images):
        distances = [mesh.outerInterocularDistance() if mesh is not None else None
        for mesh in groundTruthMeshs]            
        return distances

def getProcessor(evaluationType):
    if ("FACE_ALIGNMENT_SPARSE" in evaluationType):
        return FaceAlignmentSparseEvaluating(evaluationType)
    if ("FACE_ALIGNMENT_DENSE" in evaluationType):
        return FaceAlignmentDenseEvaluating(evaluationType)
    if ("FACE_RECONSTRUCTION" in evaluationType):
        return FaceRecontructonEvaluating(evaluationType)
    return None

class FaceRestoringEvaluating:
    i3e = Image3DExtracting()
    bd = BoundingboxDetecting()
    ie = ImageGenerator()
    uvr = UVMapRestoring()
    ie.logging = False

    errorHeaders = ["path", "error"]
    errorDistributionHeaders = ["errorLimit", "ratio"]

    evaluationProcessors = {}

    def __init__(self, logger=None, shiftZ=False):
        for type in config_evaluating.EVALUATION_TYPES:
            self.evaluationProcessors[type] = getProcessor(type)
        self.logger = logger
        self.shiftZ = shiftZ

    def log(self, message):
        if self.logger is not None:
            self.logger.log(message)

    def getEvaluateResult(self, imagePaths, orgMatPaths, predictMatPaths, batchSize, modelPath, evaluationTypeIdxs=None):
        processors = []
        if evaluationTypeIdxs == None:
            evaluationTypeIdxs = list(range(len(config_evaluating.EVALUATION_TYPES)))
        for idx in evaluationTypeIdxs:
            processors.append(self.evaluationProcessors[config_evaluating.EVALUATION_TYPES[idx]])

        normalizeMeanErrorsList = self.__gettingNormalizeMeanError(imagePaths, orgMatPaths, predictMatPaths, batchSize, modelPath, processors)
        return normalizeMeanErrorsList

    def evaluate(self, imagePaths, matPaths, batchSize, modelPath, evaluationTypeIdxs=None, hdf5Path=None, imageNum=None, zipOutput=True):
        processors = []
        if evaluationTypeIdxs == None:
            evaluationTypeIdxs = list(range(len(config_evaluating.EVALUATION_TYPES)))
        for idx in evaluationTypeIdxs:
            processors.append(self.evaluationProcessors[config_evaluating.EVALUATION_TYPES[idx]])

        self.zipOutput = zipOutput
        self.rawImages = None
        self.rawMats = None
        self.imagePaths = None
        self.matPaths = None
        if hdf5Path is not None and os.path.exists(hdf5Path):
            from inout.hdf5datasetreader import HDF5DatasetReader
            dbReader = HDF5DatasetReader(hdf5Path)
            dbImage = dbReader.image
            dbUVMap = dbReader.uvmap
            dbUVMapSize = dbReader.uvmapSize[:imageNum]            
            self.rawImages = [bytes(image) for image in dbImage[:imageNum]]
            self.rawMats = [bytes(mat)[:size] for (mat, size) in zip(dbUVMap[:], dbUVMapSize[:imageNum])]
            self.imageNum = len(self.rawImages)
        else:
            self.imagePaths = imagePaths[:imageNum]
            self.matPaths = matPaths[:imageNum]
            self.imageNum = len(self.matPaths)

        self.jsonStatusPath = config_evaluating.EVALUATION_STATUS_JSON_PATH
        self.evalStatus = {}
        self.processedNum = 0
        if (os.path.exists(self.jsonStatusPath)):
            self.evalStatus = json.loads(open(config_evaluating.EVALUATION_STATUS_JSON_PATH).read())
            self.processedNum = self.evalStatus["processedNum"]

        self.__writingNormalizeMeanError(batchSize, modelPath, processors)
        self.__writingCumulativeErrorsDistribution(processors)
        self.__drawingCumulativeErrorsDistributionFig(processors, evaluationTypeIdxs)

    def __gettingNormalizeMeanError(self, imagePaths, orgMatPaths, predictMatPaths, batchSize, modelPath, processors):
        print("[INFO] Calculating and writing normalize mean error...")
        model = modelPath
        if isinstance(modelPath, str) and predictMatPaths is None:
            from tensorflow.keras.models import load_model
            model = load_model(modelPath, compile=False)

        normalizeMeanErrorsList = []
        pbar = tqdm(desc="Processing", total=len(imagePaths))
        for i in range(0, len(imagePaths), batchSize):
            batchImagePath = imagePaths[i:i+batchSize]
            batchOrgMatPath = orgMatPaths[i:i+batchSize]
            images = [file_methods.readImage(path) if isinstance(path, str) else path for path in batchImagePath]
            orgMats = [file_methods.readMat(path) if isinstance(path, str) else path for path in batchOrgMatPath]

            # get ground truth meshs
            groundTruthMeshs = self.__getGroundTruthMesh(orgMats)
            
            # get restoring meshs
            if (predictMatPaths is None):
                restoreMeshs = self.__getRestoreMesh(images, orgMats, model)
            else:
                batchPredictMatPath = predictMatPaths[i:i+batchSize]
                predictMats = [file_methods.readMat(path) if isinstance(path, str) else path for path in batchPredictMatPath]
                restoreMeshs = self.__getRestoreMesh(None, predictMats, None)

            # get normalized sizes
            allNormalizedSizes = [processor.getNormalizedSizes(groundTruthMeshs, images)
                for processor in processors]

            # get mesh info pairs for calculating
            allMeshPairs = [processor.getMeshPairs(groundTruthMeshs, restoreMeshs) for processor in processors]
            allPossPairs = [processor.getPositionPairs(meshPairs[0], meshPairs[1]) 
                for (processor, meshPairs) in zip(processors, allMeshPairs)]
            allNormalizeMeanErrors = [self.__calculateNormalizeError(possPairs[0], 
                possPairs[1], normalizedSizes, processor.mode)
                for (processor, possPairs, normalizedSizes) in zip(processors, allPossPairs, allNormalizedSizes)]
            allNormalizeMeanErrors = [list(errors) for errors in zip(*allNormalizeMeanErrors)]
            normalizeMeanErrorsList.extend(allNormalizeMeanErrors)

            pbar.update(len(batchImagePath))

        return normalizeMeanErrorsList

    def __writingNormalizeMeanError(self, batchSize, modelPath, processors):
        print("[INFO] Calculating and writing normalize mean error...")
        model = modelPath
        if isinstance(modelPath, str):
            from tensorflow.keras.models import load_model
            model = load_model(modelPath, compile=False)

        writeAppend = True if self.processedNum > 0 else False
        [processor.initializeCsvFile("NME", "write", writeAppend) for processor in processors]
        
        if writeAppend is False:
            [processor.writerow(self.errorHeaders) for processor in processors]

        pbar = tqdm(desc="Writing", total=self.imageNum, initial=self.processedNum)
        self.log("Writing {} -> {}...".format(self.processedNum, self.imageNum))
        for i in range(self.processedNum, self.imageNum, batchSize):
            if self.rawImages is not None:
                batchImagePath = self.rawImages[i:i+batchSize]
                batchImagePathName = ["{0:04d}".format(i) for i in range(i, i+len(batchImagePath))]
            else:
                batchImagePath = self.imagePaths[i:i+batchSize]
                batchImagePathName = batchImagePath

            if self.rawMats is not None:
                batchMatPath = self.rawMats[i:i+batchSize]
            else:            
                batchMatPath = self.matPaths[i:i+batchSize]

            images = [file_methods.readImage(path) for path in batchImagePath]
            mats = [file_methods.readMat(path) for path in batchMatPath]

            # get ground truth meshs
            groundTruthMeshs = self.__getGroundTruthMesh(mats)
            
            # get restoring meshs
            restoreMeshs = self.__getRestoreMesh(images, mats, model)

            # get normalized sizes
            allNormalizedSizes = [processor.getNormalizedSizes(groundTruthMeshs, images)
                for processor in processors]

            # get mesh info pairs for calculating
            allMeshPairs = [processor.getMeshPairs(groundTruthMeshs, restoreMeshs) for processor in processors]
            allPossPairs = [processor.getPositionPairs(meshPairs[0], meshPairs[1]) 
                for (processor, meshPairs) in zip(processors, allMeshPairs)]
            allNormalizeMeanErrors = [self.__calculateNormalizeError(possPairs[0], 
                possPairs[1], normalizedSizes, processor.mode)
                for (processor, possPairs, normalizedSizes) in zip(processors, allPossPairs, allNormalizedSizes)]
            [processor.writerows(zip(batchImagePathName, normalizeMeanErrors)) 
                for (processor, normalizeMeanErrors) in zip(processors, allNormalizeMeanErrors)]
            [processor.flush() for processor in processors]

            self.evalStatus["processedNum"] = i + len(batchImagePath)
            f = open(self.jsonStatusPath, "w")
            f.write(json.dumps(self.evalStatus))
            f.close()

            if (self.zipOutput):
                outputDir = config_evaluating.EVALUATION_PATH
                ouputZipDir = file_methods.getParentPath(outputDir)
                ouputZipName = file_methods.getFileName(outputDir)
                ouputZipPath = os.path.sep.join([ouputZipDir, ouputZipName])
                shutil.make_archive(ouputZipPath, 'zip', outputDir)
                self.log(ouputZipPath+".zip")

            pbar.update(len(batchImagePath))

            dict = pbar.format_dict
            remaining = ((dict["total"] - dict["n"]) / dict["rate"]
                if dict["rate"] and dict["total"] else 0)
            remaining_str = tqdm.format_interval(remaining) if dict["rate"]  else '?'            
            elapsed_str = tqdm.format_interval(dict["elapsed"])
            self.log("Processed {0}/{1} [{2}<{3}]".format(
                i+len(batchImagePath), self.imageNum,
                elapsed_str, remaining_str))

        [processor.closeCsvFile("NME", "write") for processor in processors]
        pbar.close()

    def __writingCumulativeErrorsDistribution(self, processors):
        print("[INFO] Calculating and writing cumulative errors distribution...")
        for processor in processors:
            print("Processing for {}...".format(processor.evaluationInfo.FILEPATH_CED))
            processor.initializeCsvFile("NME", "read")
            errors = [row for row in processor.readerNME]
            errors = errors[1:]
            (_, errors) = zip(*errors)
            errors = np.array(errors).astype("float32")
            processor.closeCsvFile("NME", "read")

            processor.initializeCsvFile("CED", "write")
            processor.writerCED.writerow(self.errorDistributionHeaders)
            
            errorLimits = np.linspace(0,1,1000)
            errorRatios = [(errors < errorLimits[i]).sum()/float(len(errors)) for i in range(1000)]
            errorMeans = [errors[errors < errorLimits[i]].mean() for i in range(1000)]

            processor.writerCED.writerows(zip(errorLimits, errorRatios, errorMeans))

            processor.closeCsvFile("CED", "write")

    def __drawingCumulativeErrorsDistributionFig(self, processors, evaluationTypeIdxs):
        print("[INFO] Drawing cumulative errors distribution...")
        allErrorLimits = []
        allErrorRatios = []
        allLineStyles = []
        allTypes = []
        allErrorMeanRatios = []
        allFigPaths = []
        for processor in processors:
            print("Getting values for {}...".format(processor.evaluationInfo.FIGPATH_CED))
            processor.initializeCsvFile("CED", "read")
            errorDistributions = [row for row in processor.readerCED]
            errorDistributions = errorDistributions[1:]
            (errorLimits, errorRatios, errorMeans) = zip(*errorDistributions)
            errorLimits = np.array(errorLimits).astype("float32")
            errorRatios = np.array(errorRatios).astype("float32")
            errorMeanRatio = float(errorMeans[-1]) * 100
            processor.closeCsvFile("CED", "read")

            allErrorLimits.append(errorLimits)
            allErrorRatios.append(errorRatios)
            allErrorMeanRatios.append(errorMeanRatio)
            allLineStyles.append(processor.evaluationInfo.LINE_STYLE)
            allTypes.append(processor.evaluationInfo.TYPE_DISPLAY_NAME.lower())
            allFigPaths.append(processor.evaluationInfo.FIGPATH_CED)
        
        for i, type in enumerate(allTypes):
            print("Drawing values for {}...".format(processor.evaluationInfo.FIGPATH_CED))
            errorLimits = allErrorLimits[i]
            errorRatios = allErrorRatios[i]
            errorMeanRatio = allErrorMeanRatios[i]
            lineStyle = allLineStyles[i]
            figPath = allFigPaths[i]

            plt.clf()
            plt.xlim(0,7)
            plt.ylim(0,100)
            plt.yticks(np.arange(0,110,10))
            plt.xticks(np.arange(0,11,1))
            plt.grid()
            plt.title('NME (%)', fontsize=20)
            plt.xlabel('NME (%)', fontsize=16)
            plt.ylabel('Test Images (%)', fontsize=16)
            plt.plot(errorLimits*100, errorRatios*100, lineStyle,
                label='PRN ({}): {:.2f}'.format(type, errorMeanRatio), 
                lw=3)
            plt.legend(loc=4, fontsize=10)
            plt.savefig(figPath)

        print("Drawing values for {}...".format(config_evaluating.EVALUATION_FIGPATH_CED))
        plt.clf()
        plt.xlim(0,7)
        plt.ylim(0,100)
        plt.yticks(np.arange(0,110,10))
        plt.xticks(np.arange(0,11,1))
        plt.grid()
        plt.title('NME (%)', fontsize=20)
        plt.xlabel('NME (%)', fontsize=16)
        plt.ylabel('Test Images (%)', fontsize=16)
        for i, type in enumerate(allTypes):
            errorLimits = allErrorLimits[i]
            errorRatios = allErrorRatios[i]
            errorMeanRatio = allErrorMeanRatios[i]
            lineStyle = allLineStyles[i]
            figPath = allFigPaths[i]
            plt.plot(errorLimits*100, errorRatios*100, lineStyle,
                label='PRN ({}): {:.2f}'.format(type, errorMeanRatio), 
                lw=3)
        plt.legend(loc=4, fontsize=10)

        posfix = "_".join([str(idx) for idx in evaluationTypeIdxs])
        plt.savefig(config_evaluating.EVALUATION_FORMAT_FIGPATH_CED.format(posfix))

    def __getGroundTruthMesh(self, mats):
        meshs = [self.i3e.preprocess(mat) if mat is not None else mat for mat in mats]
        for i in range(len(meshs)):
            if meshs[i] is not None:
                meshs[i].vertices[:, 2] = meshs[i].vertices[:, 2] - np.min(meshs[i].vertices[:, 2])
        return meshs
    
    def __getRestoreMesh(self, images, mats, model=None):
        if model is not None:
            matsList = [[mat] for mat in mats]
            faceImageInfos = self.ie.generateMulti(images, matsList)
            faceImageInfos = [faceImageInfo[0] for faceImageInfo in faceImageInfos]
            (_, faceImages, tforms) = zip(*faceImageInfos)
            uvmaps = model.predict(np.array(faceImages))
            uvmaps = [uvmap for uvmap in uvmaps]
            meshs = [self.uvr.postprocess(image, uvmap, tform) for (image, uvmap, tform) in zip(images, uvmaps, tforms)]
        else:
            meshs = [self.i3e.preprocess(mat) if mat is not None else mat for mat in mats]

        if (self.shiftZ):
            for i in range(len(meshs)):
                meshs[i].vertices[:, 2] = meshs[i].vertices[:, 2] - np.min(meshs[i].vertices[:, 2])
        
        return meshs

    def __calculateNormalizeError(self, groundTruthPoss, restorePoss, normalizedSizes, mode):
        noneIndexes = []
        notNoneIndexes = []
        for idx in range(len(groundTruthPoss)):
            if (groundTruthPoss[idx] is not None 
                and restorePoss[idx] is not None 
                and normalizedSizes[idx] is not None):
                notNoneIndexes.append(idx)
            else:
                noneIndexes.append(idx)
        
        groundTruthNotNonePoss = np.array([groundTruthPoss[i] for i in notNoneIndexes])
        restoreNotNonePoss = np.array([restorePoss[i] for i in notNoneIndexes])
        normalizedNotNoneSizes = np.array([normalizedSizes[i] for i in notNoneIndexes])
        
        # if (groundTruthPoss is None or restorePoss is None or normalizedSizes is None):
        #     return 0      
        distances = np.power((groundTruthNotNonePoss - restoreNotNonePoss), 2)
        if mode == "2D":
            distances = np.sqrt(np.sum(distances[:,:,:2], axis=2))
        else:
            distances = np.sqrt(np.sum(distances, axis=2))
        errorsList = distances / normalizedNotNoneSizes[:,np.newaxis]
        errorsMean = np.mean(errorsList, axis=1)

        allErrorsMean = np.zeros((len(groundTruthPoss)))
        allErrorsMean[notNoneIndexes] = errorsMean

        return allErrorsMean

if __name__ == "__main__":

    from configure import config_training

    EVALUATION_PATH = config_evaluating.BASE_EVALUATION_PATH + "_temp"
    config_evaluating.createEvaluationInfos(baseEvaluationPath=EVALUATION_PATH)


    OUTPUT_DIR = r"M:\My Drive\CaoHoc\LUANVAN\SourceCode\output_wtmse_0.0002"
    MODEL_PATH = os.path.sep.join([OUTPUT_DIR, config_training.BEST_VAL_LOSS_MODEL_FILE_NAME])


    fe = FaceRestoringEvaluating()


    imagePaths = [r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000\image00002.jpg",
        r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000\image00006.jpg"]
    matPaths = [r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000\image00002.mat",
        r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000\image00006.mat"]
    nmes = fe.getEvaluateResult(imagePaths, matPaths, None, 20, MODEL_PATH)

    print(nmes)