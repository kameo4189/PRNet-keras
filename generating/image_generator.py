import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
from configure import config
from preprocessing.image2D_warping import Image2DWarping
from detecting.face_detecting import FaceDetecting
from detecting.boundingbox_detecting import BoundingboxDetecting
from preprocessing.image3D_extracting import Image3DExtracting
import skimage
from util import file_methods
import warnings

class ImageGenerator:
    fd = FaceDetecting()
    bd = BoundingboxDetecting()
    i2w = Image2DWarping()
    i3e = Image3DExtracting()
    logging = True

    def log(self, content):
        if self.logging:
            print(content)

    def generateMulti(self, images, mats=None, outHeight=config.IMAGE_HEIGHT, outWidth=config.IMAGE_WIDTH, 
        dataTypeMode=config.DATA_TYPE_MODE_DEFAULT):
        inputImages = [file_methods.readImage(image, False) for image in images]

        self.log("Detecting face bounding box...")
        faceBBInfosList = self.fd.detectMulti(inputImages, mats)

        self.log("Warping image for face image...")
        faceImageInfosList = [[] for _ in range(len(inputImages))]
        imageDType = config.DATA_TYPE_MODES[dataTypeMode][0]
        for i, faceBBInfos in enumerate(faceBBInfosList):
            faceImageWithTforms = [self.i2w.preprocess(skimage.img_as_float(inputImages[i]), 
                faceBBInfo, outHeight, outWidth) for faceBBInfo in faceBBInfos]
            
            for faceBBInfo, (faceImage, tform) in zip(faceBBInfos, faceImageWithTforms):
                if (imageDType == "uint8"):
                    warnings.filterwarnings("ignore", category=UserWarning)
                    faceImage = skimage.img_as_ubyte(faceImage)
                    warnings.filterwarnings("default", category=UserWarning)
                else:
                    faceImage = faceImage.astype(imageDType)
                
                faceImageInfosList[i].append((faceBBInfo, faceImage, tform))

        return faceImageInfosList
    
    def generate(self, image, mat=None, outHeight=config.IMAGE_HEIGHT, outWidth=config.IMAGE_WIDTH, 
        dataTypeMode=config.DATA_TYPE_MODE_DEFAULT):
        inputImage = file_methods.readImage(image, False)

        self.log("Detecting face bounding box...")
        faceBBInfos = self.fd.detect(inputImage, mat)
        if len(faceBBInfos) == 0:
            print('[WARNING] no detected face')
            return []
        
        self.log("Warping image for face image...")
        imageDType = config.DATA_TYPE_MODES[dataTypeMode][0]
        faceImageInfos = []
        for faceBBInfo in faceBBInfos:
            (faceImage, tform) = self.i2w.preprocess(skimage.img_as_float(inputImage), 
                faceBBInfo, outHeight, outWidth)
            if (imageDType == "uint8"):
                warnings.filterwarnings("ignore", category=UserWarning)
                faceImage = skimage.img_as_ubyte(faceImage)
                warnings.filterwarnings("default", category=UserWarning)
            else:
                faceImage = faceImage.astype(imageDType)
         
            faceImageInfos.append((faceBBInfo, faceImage, tform))

        return faceImageInfos

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import skimage.transform as skt
    
    image_path = r"D:\Pictures\FB_IMG_1612923425427.jpg"
    image = skimage.io.imread(image_path)

    ig = ImageGenerator()
    faceImageInfos = ig.generate(image_path)

    for info in faceImageInfos:       
        (faceBBInfo, faceImage, tform) = info
        (center,size) = faceBBInfo
        topLeft = (center[0]-size//2, center[1]-size//2)

        plt.imshow(image)
        plt.gca().add_patch(Rectangle(topLeft,size,size,linewidth=1,edgecolor='b',facecolor='none'))
    plt.show()

    for i, info in enumerate(faceImageInfos):       
        (faceBBInfo, faceImage, tform) = info

        plt.subplot(1, len(faceImageInfos), i+1)
        plt.imshow(faceImage)
    plt.show()