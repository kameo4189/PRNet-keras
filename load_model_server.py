# import nessesary packages
import numpy as np
from util import file_methods
from util.resful_api_methods import InputData, OutputData, FaceMeshData
from util.resful_api_methods import InputEvalData, MeshEvalResult, OutputEvalData
from fastapi import FastAPI
from util import server_methods
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
    default=r"M:\My Drive\CaoHoc\LUANVAN\SourceCode\output_wtmse_0.0001\best_eval_loss.model",
    help="path to model")
args = vars(ap.parse_args())

def ConstructAIServer(sharingDict):
    app = FastAPI()
    MODEL_PATH = args["model"]

    @app.on_event("startup")
    def initializeModel():
        from tensorflow.keras.models import load_model
        from generating.image_generator import ImageGenerator
        from preprocessing.image3D_extracting import Image3DExtracting
        from postprocessing.uvmap_restoring import UVMapRestoring
        from detecting.face_detecting import FaceDetecting
        global model
        global imageGenerating
        global uvmapRestoring
        global image3DExtracting
        global faceDectecting
        global faceRestoringEvaluating

        model = load_model(MODEL_PATH, compile=False)
        imageGenerating = ImageGenerator()
        uvmapRestoring = UVMapRestoring()
        image3DExtracting = Image3DExtracting()
        faceDectecting = FaceDetecting()

        from configure import config_evaluating
        from evaluating.face_restoring_evaluating import FaceRestoringEvaluating
        EVALUATION_PATH = config_evaluating.BASE_EVALUATION_PATH + "_temp"
        config_evaluating.createEvaluationInfos(baseEvaluationPath=EVALUATION_PATH)
        faceRestoringEvaluating = FaceRestoringEvaluating(shiftZ=True)

    def readInputData(inputData: InputData):
        inputImageMats = [(imageMat.rawImage, imageMat.rawMats) for imageMat in inputData.imageMatList]
        (inputImageData, inputMatsListData) = zip(*inputImageMats)

        inputRawImages = [file_methods.stringToBytes(image) for image in inputImageData]
        inputRawMatsList = [[file_methods.stringToBytes(mat) for mat in mats] for mats in inputMatsListData]
        inputImages = [file_methods.readImage(image, False) for image in inputRawImages]
        inputImages = [np.repeat(image[:,:,None], 3, axis=2) if (len(image.shape) == 2) else image for image in inputImages]
        inputMatsList = [[file_methods.readMat(mat) if len(mat)>0 else None for mat in mats] for mats in inputRawMatsList]

        return inputImages, inputMatsList

    def readInputEvalData(inputData: InputEvalData):
        inputImageMats = [(imageMat.rawImage, imageMat.rawOrgMats, imageMat.rawPredictMats) for imageMat in inputData.imageMatList]
        (inputImageData, inputMatsOrgListData, inputMatsPredictListData) = zip(*inputImageMats)

        inputRawImages = [file_methods.stringToBytes(image) for image in inputImageData]
        inputRawMatsOrgList = [[file_methods.stringToBytes(mat) for mat in mats] for mats in inputMatsOrgListData]
        inputRawMatsPredictList = [[file_methods.stringToBytes(mat) for mat in mats] for mats in inputMatsPredictListData]
        inputImages = [file_methods.readImage(image, False) for image in inputRawImages]
        inputImages = [np.repeat(image[:,:,None], 3, axis=2) if (len(image.shape) == 2) else image for image in inputImages]
        
        inputMatsOrgList = [[file_methods.readMat(mat) if len(mat)>0 else None for mat in mats] for mats in inputRawMatsOrgList]
        inputMatsPredictList = [[file_methods.readMat(mat) if len(mat)>0 else None for mat in mats] for mats in inputRawMatsPredictList]

        return inputImages, inputMatsOrgList, inputMatsPredictList

    @app.get('/')
    def index():
        return {'message': 'This is the homepage of the PRNet API'}

    @app.post('/evaluate')
    def evaluate(data: InputEvalData):
        print("[INFO] Reading images and mats...")
        inputImages, inputMatsOrgList, inputMatsPredictList = readInputEvalData(data)
        
        evalImages = []
        evalMatsOrg = []
        evalMatsPredict = []
        evalIndexes = []
        for i in range(len(inputImages)):
            evalIndexes.extend([i for _ in range(len(inputMatsPredictList[i]))])
            evalImages.extend([inputImages[i] for _ in range(len(inputMatsPredictList[i]))])
            evalMatsOrg.extend(inputMatsOrgList[i])
            evalMatsPredict.extend(inputMatsPredictList[i])

        print("[INFO] Evaluating mesh info...")
        nmesList = faceRestoringEvaluating.getEvaluateResult(inputImages, evalMatsOrg, evalMatsPredict, 20,
            model)
        nmesList = [[float(nme) for nme in nmes] for nmes in nmesList]
        
        print("[INFO] Creating response data...")
        response = OutputEvalData(numImages=len(inputImages))
        for idx in range(len(nmesList)):
            imageIdx = evalIndexes[idx]
            response.evaluationDataList[imageIdx].imageEvalValues.append(MeshEvalResult())
            response.evaluationDataList[imageIdx].imageEvalValues[-1].evalValues = nmesList[idx]

        print("[INFO] Returning response data...")
        return response

    @app.post('/extract')
    def extract(data: InputData):
        print("[INFO] Reading images and mats...")
        inputImages, inputMatsList = readInputData(data)
        
        print("[INFO] Extracting mesh info...")
        meshInfosList = [[image3DExtracting.preprocess(mat) if mat is not None else None for mat in inputMats] for inputMats in inputMatsList]
        rawMeshMatsList = [[mesh.toCommonFaceRegionMesh().toBytes(data.matType, np.float32, np.int32) 
            if mesh is not None else None for mesh in meshInfos] for meshInfos in meshInfosList]
        rawMeshMatsList = [[file_methods.bytesToString(mesh) if mesh is not None else "" for mesh in rawMeshMats] for rawMeshMats in rawMeshMatsList]
        faceBBInfosList = faceDectecting.detectMulti(inputImages, inputMatsList, onlyMats=True)

        print("[INFO] Creating response data...")
        response = OutputData(numImages=len(inputImages))
        for imageIdx, (faceBBs, rawMeshMats) in enumerate(zip(faceBBInfosList, rawMeshMatsList)):
            for i in range(len(faceBBs)):
                rawMeshMat = rawMeshMats[i]
                bb = []
                if faceBBs[i] is not None:
                    (centerX, centerY), size = faceBBs[i]
                    centerX, centerY, size = int(centerX), int(centerY), int(size)
                    bb = [centerX, centerY, size]
                response.imageMeshList[imageIdx].faceMeshList.append(FaceMeshData(rawMesh=rawMeshMat, faceBB=bb))
        
        print("[INFO] Returning response data...")
        return response

    @app.post('/predict')
    def predict(data: InputData):
        print("[INFO] Reading images and mats...")
        inputImages, inputMatsList = readInputData(data)

        print("[INFO] Creating input face images...")
        faceInfosList = imageGenerating.generateMulti(inputImages, inputMatsList)
        faceInfosIndexes = list(range(len(faceInfosList)))
        faceInfosListWithIndex = zip(faceInfosIndexes, inputImages, faceInfosList)
        faceImageWithIndexes = [(index, image, *faceInfo) 
            for (index, image, faceInfos) in faceInfosListWithIndex 
            for faceInfo in faceInfos]
        (indexes, orgImages, faceBBs, faceImages, tforms) = zip(*faceImageWithIndexes)

        print("[INFO] Predicting for uv position maps...")
        uvmaps = model.predict(np.array(faceImages))
        uvmaps = [uvmap for uvmap in uvmaps]

        # restoring mesh from uv position maps
        print("[INFO] Restoring mesh info...")
        restoredMeshInfos = [uvmapRestoring.postprocess(image, uvmap, tform) for 
            (image, uvmap, tform) in zip(orgImages, uvmaps, tforms)]
        rawMeshMats = [mesh.toBytes(data.matType, np.float32, np.int32) for mesh in restoredMeshInfos]
        rawMeshMats = [file_methods.bytesToString(mesh) for mesh in rawMeshMats]
        
        print("[INFO] Creating response data...")
        response = OutputData(numImages=len(inputImages))
        for i, faceBB, rawMeshMat in zip(indexes, faceBBs, rawMeshMats):
            (centerX, centerY), size = faceBB
            centerX, centerY, size = int(centerX), int(centerY), int(size)
            bb = [centerX, centerY, size]
            response.imageMeshList[i].faceMeshList.append(FaceMeshData(rawMesh=rawMeshMat, faceBB=bb))
        
        print("[INFO] Returning response data...")
        return response

    server_methods.startServer(app, sharingDict=sharingDict, port=20000, code=False)

if __name__ == "__main__":
    server_methods.autoRestartServerProcess(ConstructAIServer, print, 3*60)