import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
from evaluating.face_alignment_dense_evaluating import FaceAlignmentDenseEvaluating

class FaceRecontructionEvaluating(FaceAlignmentDenseEvaluating):

    lineStyle = "y-"
    evaluateMode = "reconstruct"

    def __getMeshPair(self, groundTruthMeshs, restoreMeshs):
        interative_closest_point.previewing = False
        interative_closest_point.logging = False
        meshPairs = [self.alignMesh(gtMesh, restoredMesh) for gtMesh, restoredMesh 
            in zip(groundTruthMeshs, restoreMeshs)]
        (alignedMeshs, fixMeshs) = zip(*meshPairs)       
        return (alignedMeshs, fixMeshs)

    def alignMesh(self, fixMesh, moveMesh):
        X_fix = fixMesh.vertices
        X_move = moveMesh.vertices
        
        X_fix_mean = np.mean(X_fix, axis=0)
        X_move_mean = np.mean(X_move, axis=0)
        translations = X_move_mean - X_fix_mean
        X_move = X_move - translations

        (_, X_move_transformed, nearest_idxs) = interative_closest_point.simpleicp(X_fix, X_move, X_move.shape[0]//2, min_change=0.1)

        tranformedMoveMesh = moveMesh.copy()
        tranformedMoveMesh.vertices = X_move_transformed
        nearestFixMesh = fixMesh.copy()
        nearestFixMesh.vertices = nearestFixMesh.vertices[nearest_idxs]
        nearestFixMesh.colors = nearestFixMesh.colors[nearest_idxs]
        nearestFixMesh.triangles = None
        return tranformedMoveMesh, nearestFixMesh

if __name__ == "__main__":
    import argparse
    from tensorflow.keras.models import load_model
    from generating.image_generator import ImageGenerator
    from util import interative_closest_point, file_methods
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
        default = r"K:\Study\CaoHoc\LuanVan\dataset\AFLW2000\image00040.jpg",
        help="path to image")
    ap.add_argument("-m", "--model", type=str,
        default=r"E:\My Drive\CaoHoc\LUANVAN\SourceCode\output_wtrmse_0.01\best_eval_loss.model",
        help="path to model")
    args = vars(ap.parse_args())

    IMAGE_PATH = args["image"]
    MODEL_PATH = args["model"]

    print("[INFO] Reading image...")
    image = file_methods.readImage(IMAGE_PATH, False)

    print("[INFO] Creating input face image...")
    ig = ImageGenerator()
    matPath = file_methods.getFilePathWithOtherExt(IMAGE_PATH, ".mat")
    restoreMeshPath = file_methods.getFilePathWithOtherParent(matPath, r"data")
    
    from preprocessing.image3D_extracting import Image3DExtracting
    i3e = Image3DExtracting()
    gtMeshInfo = i3e.preprocess(matPath)

    if os.path.exists(restoreMeshPath) is False:
        faceInfoList = ig.generate(image, matPath)
        if (len(faceInfoList) == 0):
            print("[WARNING] No face detected")
            exit()

        print("[INFO] Loading pre-trained network...")
        model = load_model(MODEL_PATH, compile=False)

        print("[INFO] Predicting for uv position map...")
        faceBBs, faceImages, tforms = zip(*faceInfoList)
        uvmaps = model.predict(np.array(faceImages))
        uvmaps = [uvmap for uvmap in uvmaps]
        uvmap = uvmaps[0]
        tform = tforms[0]

        print("[INFO] Reconstructing for mesh...")
        from postprocessing.uvmap_restoring import UVMapRestoring
        uvr = UVMapRestoring()
        restoredMeshInfo = uvr.postprocess(image, uvmap, tform)
        restoredMeshInfo.save(restoreMeshPath)
    else:
        restoredMeshInfo = i3e.preprocess(restoreMeshPath)
       
    from util import mesh_display
    mesh_display.displayMeshPointCloud([gtMeshInfo, restoredMeshInfo])
    gtMeshInfo.vertices[:, 2] = gtMeshInfo.vertices[:, 2] - np.min(gtMeshInfo.vertices[:, 2])
    fre = FaceRecontructionEvaluating()
    interative_closest_point.previewing = True
    interative_closest_point.logging = True
    tranformedMesh, nearestGTMesh = fre.alignMesh(gtMeshInfo, restoredMeshInfo)
    nearestGTMeshPreview = nearestGTMesh.copy()
    nearestGTMeshPreview.vertices = nearestGTMeshPreview.vertices + 200
    mesh_display.displayMeshPointCloud([nearestGTMesh, nearestGTMeshPreview, tranformedMesh])