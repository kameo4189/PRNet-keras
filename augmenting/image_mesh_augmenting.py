import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from configure import config_augmenting as confa
from preprocessing.geometric_transforming import GeometricTransforming
from preprocessing.color_scaling import ColorScaling
from preprocessing.cutout_random_erasing import CutoutRandomErasing
from preprocessing.color_jittering import ColorJittering
import numpy as np
from numpy.random import uniform, shuffle

class ImageMeshAugmenting:
    cre = CutoutRandomErasing()
    gt = GeometricTransforming()
    cj = ColorJittering()

    crePerformedRatio = confa.RANDOM_ERASING_PERFORM_RATIO
    (creAreaRatioMin, creAreaRatioMax) = confa.RANDOM_ERASING_AREA_RATIO_RANGE
    (creAspectRatioMin, creAspectRatioMax) = confa.RANDOM_ERASING_ASPECT_RATIO_RANGE
    (creAreaValueMin, creAreaValueMax) = confa.RANDOM_ERASING_AREA_VALUE_RANGE
    crePixelLevel = confa.RANDOM_ERASING_PIXEL_LEVEL
    creGrayArea = confa.RANDOM_ERASING_GRAY_AREA

    (rotMin, rotMax) = confa.TRANSFORMING_ROTATION_RANGE
    (transMin, transMax) = confa.TRANSFORMING_TRANSLATION_RATIO_RANGE
    (scaleMin, scaleMax) = confa.TRANSFORMING_SCALE_RATIO_RANGE

    (colorScaleMin, colorScaleMax)= confa.COLOR_SCALE_RATIO_RANGE
    (colorBrightnessMin, colorBrightnessMax)= confa.COLOR_JITTER_BRIGHTNESS_RANGE
    (colorContrastMin, colorContrastMax)= confa.COLOR_JITTER_CONTRAST_RANGE
    (colorSaturationMin, colorSaturationMax)= confa.COLOR_JITTER_SATURATION_RANGE
    (colorGammaMin, colorGammaMax)= confa.COLOR_JITTER_GAMMA_RANGE
    colorBrightnessPerform = confa.COLOR_JITTER_BRIGHTNESS_PERFORM
    colorContrastPerform = confa.COLOR_JITTER_CONTRAST_PERFORM
    colorSaturationPerform = confa.COLOR_JITTER_SATURATION_PERFORM
    colorGammaPerform = confa.COLOR_JITTER_GAMMA_PERFORM

    borderMode="reflect"

    def augment(self, image, meshInfo):
        (w, h) = (image.shape[1], image.shape[0])

        # random erasing agumentation
        augImage = self.cre.preprocess(image, self.crePerformedRatio, self.creAreaRatioMin, self.creAreaRatioMax,
            self.creAspectRatioMin, self.creAspectRatioMax, self.creAreaValueMin, self.creAreaValueMax,
            self.crePixelLevel, self.creGrayArea)

        # geometric transform agumentation
        rotation = uniform(self.rotMin, self.rotMax)
        translationX = uniform(self.transMin, self.transMax)
        translationY = uniform(self.transMin, self.transMax)
        translations = (translationX * w, translationY * h)
        scale = uniform(self.scaleMin, self.scaleMax)
        (augImage, augMeshInfo, tform) = self.gt.preprocess(augImage, meshInfo, scale, rotation,
            translations, self.borderMode)
        
        # color jitter agumentation
        brightness = 1 if self.colorBrightnessPerform == False else uniform(self.colorBrightnessMin, self.colorBrightnessMax)
        contrast = 1 if self.colorContrastPerform == False else uniform(self.colorContrastMin, self.colorContrastMax)
        saturation = 1 if self.colorSaturationPerform == False else uniform(self.colorSaturationMin, self.colorSaturationMax)
        gamma = 1 if self.colorGammaPerform == False else uniform(self.colorGammaMin, self.colorGammaMin)
        colorScale = uniform(self.colorScaleMin, self.colorScaleMax)
        adjustOrder = np.arange(5)
        np.random.shuffle(adjustOrder)
        augImage = self.cj.preprocess(augImage, brightness, contrast, saturation, gamma, colorScale, adjustOrder) 
        return (augImage, augMeshInfo, tform)

    def augments(self, images, meshInfos):
        assert len(images) == len(meshInfos), "Size of image list isn't equal mesh list"
        augmentData = [self.augment(image, meshInfo) for image, meshInfo in list(zip(images, meshInfos))]
        (augImages, augMeshInfos, tforms) = zip(*augmentData)
        return (augImages, augMeshInfos, tforms)

if __name__ == "__main__":
    import numpy as np
    from skimage import io
    import matplotlib.pyplot as plt
    from preprocessing.image3D_extracting import Image3DExtracting
    from preprocessing.image3D_to_2D import Image3DTo2D
    from preprocessing.image_mesh_warping import ImageMeshWarping
    from util import file_methods

    dataPath = r"K:\Study\CaoHoc\LuanVan\dataset\300W_LP\IBUG"
    imageWithMatPaths = file_methods.getImageWithMatList(dataPath)
    np.random.shuffle(imageWithMatPaths)
    # (imagePaths, matPaths) = zip(*imageWithMatPaths)

    for imageMat in imageWithMatPaths:
        (image_path, mat_path) = imageMat
            
        image2D = io.imread(image_path)/255.
        [h, w, c] = image2D.shape

        ia = ImageMeshAugmenting()
        i3t2 = Image3DTo2D()
        i3e = Image3DExtracting()

        meshInfo = i3e.preprocess(mat_path)

        imw = ImageMeshWarping()
        (wrappedImage, wrappedMesh, tform) = imw.preprocess(image2D, meshInfo)

        [h, w, c] = wrappedImage.shape
        (augImage, augMeshInfo, tform) = ia.augment(wrappedImage, wrappedMesh)
        transformedImageFace = i3t2.preprocess(augMeshInfo, h, w, False)
        transformedImageFaceKpts = i3t2.drawKeypoints(transformedImageFace, augMeshInfo)

        (h,w) = (augImage.shape[0], augImage.shape[1])
        plt.subplot(3, 1, 1)
        plt.imshow(image2D)
        plt.subplot(3, 1, 2)
        stackImage = np.concatenate((wrappedImage, augImage), axis=1)
        plt.imshow(stackImage)
        plt.subplot(3, 1, 3)
        plt.imshow(augImage)
        plt.imshow(transformedImageFaceKpts, alpha=0.5)
        plt.show(block=True)


