import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from configure import config
import numpy as np
import dlib
from detecting.boundingbox_detecting import BoundingboxDetecting
from preprocessing.image3D_extracting import Image3DExtracting
from util import file_methods

class FaceDetecting:
    cnn_face_detector = dlib.cnn_face_detection_model_v1(config.DLIB_DETECTION_MODEL_PATH)
    bd = BoundingboxDetecting()
    i3e = Image3DExtracting()

    def detectMulti(self, images, mats=None, normalBB=False, expandRatio=config.ORG_BB_EXTENDED_RATIO, onlyMats=False):
        inputData = images
        inputMatData = [None] * len(images)
        if (mats is not None):
            inputMatData = mats
            diff = len(images) - len(inputMatData)
            if (diff < 0):
                inputMatData = inputMatData[:-diff]
            elif (diff > 0):
                for _ in range(len(inputMatData), diff):
                    inputMatData.append(None)
            
            inputMatData = [[mats] if isinstance(mats, list) is False and mats is None else mats for mats in inputMatData]
            inputMatData = [None if mats == [None] else mats for mats in inputMatData]

        dataIndexes = list(range(len(inputData)))
        inputDataWithIndex = list(zip(dataIndexes, inputData, inputMatData))
        
        dataWithMatNone = [(index, image, mat) for (index, image, mat) in inputDataWithIndex if mat is None]
        dataWithMatNotNone = [(index, image, mat) for (index, image, mat) in inputDataWithIndex if mat is not None]
        detectedFaceBBs = []

        # processing for image with available mat
        if (len(dataWithMatNotNone) > 0):
            (notNoneMatIndexes, imageData, matData) = zip(*dataWithMatNotNone)
            meshInfosList = [[self.i3e.preprocess(mat) for mat in mats] for mats in matData]
            detectedFacesList = [[self.bd.detect(meshInfo.vertices, image.shape[0], image.shape[1], 
                randomPertube=False, normalBB=True) for (image, meshInfo) in zip(imageData, meshInfos)]
                for meshInfos in meshInfosList]
            detectedFaceBBs = list(zip(notNoneMatIndexes, detectedFacesList))

        # processing for image with not available mat 
        if (len(dataWithMatNone) > 0):
            (noneMatIndexes, imageData, _) = zip(*dataWithMatNone)
            if (onlyMats is False):                
                detectedFacesList = self._dlib_detect(list(imageData))
                for i, detected_faces in zip(noneMatIndexes, detectedFacesList):        
                    faces = [(faceRect.rect.left(), faceRect.rect.right(), faceRect.rect.top(), faceRect.rect.bottom())
                        for faceRects in detected_faces for faceRect in faceRects]               
                    detectedFaceBBs.append((i, faces))
            else:
                for i in noneMatIndexes:
                    detectedFaceBBs.append((i, None))

        bbInfoList = [[] for _ in range(len(inputData))]
        for i, bbs in detectedFaceBBs:
            (h, w) = (inputData[i].shape[0], inputData[i].shape[1])
            if bbs is not None:
                bbInfos = [self.bd.detect(np.array([[left, top], [right, bottom]]), h, w, 
                    randomPertube=False, expandRatio=expandRatio, normalBB=normalBB) 
                    for (left, right, top, bottom) in bbs]
                bbInfoList[i].extend(bbInfos)
            else:
                bbInfoList[i].append(None)

        return bbInfoList

    def detect(self, image, mat=None, normalBB=False):
        [h, w, _] = image.shape
        if ((mat is not None and isinstance(mat, str) is False) or 
            (isinstance(mat, str) and os.path.exists(mat) is True)):
            meshInfo = self.i3e.preprocess(mat)
            detected_faces = [self.bd.detect(meshInfo.vertices, h, w, 
                randomPertube=False, normalBB=True)]
        else:
            detected_faces = self._dlib_detect(image)
            detected_faces = [(face.rect.left(), face.rect.right(), face.rect.top(), face.rect.bottom())
                for face in detected_faces]

        bbInfos = []
        for detected_face in detected_faces:
            left, right, top, bottom = detected_face
            points = np.array([[left, top], [right, bottom]])
            bbInfo = self.bd.detect(points, h, w, randomPertube=False, normalBB=normalBB)  
            bbInfos.append(bbInfo)

        return bbInfos

    def _dlib_detect(self, image, batch_size=32):
        if isinstance(image, list):
            from itertools import groupby
            imageIndexes = list(range(len(image)))
            imageWithIndex = list(zip(imageIndexes, image))
            sizeGroups = groupby(imageWithIndex, lambda a: (a[1].shape[0], a[1].shape[1]))
            faceBBs = []
            for _, group in sizeGroups:
                (indexes, images) = zip(*group)
                bbInfos = self.cnn_face_detector(list(images), 1, batch_size=batch_size)
                faceBBs.extend(list(zip(indexes, bbInfos)))

            bbInfoList = [[] for _ in range(len(image))]
            for (i, bb) in faceBBs:
                bbInfoList[i].append(bb)
            return bbInfoList
        else:
            return self.cnn_face_detector(image, 1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage import io

    # image_path = r'data/IBUG_image_008_1_0.jpg'
    # image_path = r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000\image00036.jpg"
    image_path = [r"D:\Pictures\test\IMG_20210526_133801.jpg",
        #r"K:\Study\CaoHoc\LuanVan\dataset\300W_LP\HELEN\HELEN_2908549_1_4.jpg",
        r"D:\Pictures\test\FB_IMG_1612923425427.jpg",
        #r"K:\Study\CaoHoc\LuanVan\dataset\300W_LP\HELEN\HELEN_1629243_1_14.jpg",
        #r"D:\Pictures\FB_IMG_1612923425427.jpg"
        ]

    fd = FaceDetecting()
    if (isinstance(image_path, list)):
        images = [file_methods.readImage(image, False) for image in image_path]
        matPaths = [file_methods.getFilePathWithOtherExt(path, ".mat") for path in image_path]
        mats = [file_methods.readMat(path) for path in matPaths]
        bbInfoList = fd.detectMulti(images, mats, expandRatio=0.1)

        for i, (image, bbInfos) in enumerate(zip(images, bbInfoList)):
            plt.subplot(len(images), 1, i+1)
            plt.imshow(image)
            for bbInfo in bbInfos:
                (center,size) = bbInfo
                topLeft = (center[0]-size//2, center[1]-size//2)
                plt.gca().add_patch(Rectangle(topLeft,size,size,linewidth=1,edgecolor='b',facecolor='none'))
        plt.show(block=True)
    else:
        image = io.imread(image_path)
        [h, w, c] = image.shape

        bbInfo = fd.detect(image)

        (center,size) = bbInfo
        topLeft = (center[0]-size//2, center[1]-size//2)
        
        plt.imshow(image)
        plt.gca().add_patch(Rectangle(topLeft,size,size,linewidth=1,edgecolor='b',facecolor='none'))
        plt.show(block=True)