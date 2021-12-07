import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
from PIL import Image, ImageEnhance
from preprocessing.color_scaling import ColorScaling

class ColorJittering:

    def preprocess(self, image, brightness=1, contrast=1, saturation=1, gamma=1, scale=1, adjustOrder=[0,1,2,3,4]):
        pilImage = Image.fromarray(np.uint8(image*255))

        adjustFunctions = [ColorJittering.adjust_brightness,
            ColorJittering.adjust_contrast,
            ColorJittering.adjust_saturation,
            ColorJittering.adjust_gamma,
            ColorJittering.scale_color]
        params = [brightness, contrast, saturation, gamma, scale]
        
        for i in adjustOrder:
            pilImage = adjustFunctions[i](pilImage, params[i])

        return np.array(pilImage) / 255.

    @staticmethod
    def scale_color(image, scale_factor):
        if isinstance(image, Image.Image):
            scaledImage = np.array(image) / 255.
        else:
            scaledImage = image.copy()
        scaledImage = scaledImage * scale_factor
        scaledImage = np.clip(scaledImage, 0, 1)
        return Image.fromarray(np.uint8(scaledImage*255))

    @staticmethod
    def adjust_brightness(image, brightness_factor):
        if brightness_factor == 1:
            return image
        enhancer = ImageEnhance.Brightness(image)
        img = enhancer.enhance(brightness_factor)
        return img

    @staticmethod
    def adjust_contrast(image, contrast_factor):
        if contrast_factor == 1:
            return image
        enhancer = ImageEnhance.Contrast(image)
        img = enhancer.enhance(contrast_factor)
        return img

    @staticmethod
    def adjust_saturation(image, saturation_factor):
        if saturation_factor == 1:
            return image
        enhancer = ImageEnhance.Color(image)
        img = enhancer.enhance(saturation_factor)
        return img

    @staticmethod
    def adjust_gamma(image, gamma, gain=1):
        if gamma == 1:
            return image
        if gamma < 0:
            raise ValueError('Gamma should be a non-negative real number')

        input_mode = image.mode
        img = image.convert('RGB')
        gamma_map = [(255 + 1 - 1e-3) * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img = img.convert(input_mode)
        return img

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io
    from numpy.random import uniform
    from util import file_methods

    cs = ColorJittering()

    dataPath = r"K:\Study\CaoHoc\LuanVan\Dataset\AFLW2000"
    imageWithMatPaths = file_methods.getImageWithMatList(dataPath)
    np.random.shuffle(imageWithMatPaths)
    # (imagePaths, matPaths) = zip(*imageWithMatPaths)

    for imageMat in imageWithMatPaths:
        (image_path, mat_path) = imageMat
        image2D = io.imread(image_path)/255.
        [h, w, c] = image2D.shape

        brightness = uniform(0.6, 1.4)
        contrast = uniform(0.6, 1.4)
        saturation = uniform(0.6, 1.4)
        gamma = uniform(0.6, 1.4)
        scale = uniform(0.6, 1.4)
        adjustOrder = np.arange(5)
        np.random.shuffle(adjustOrder)
        transformedImage = cs.preprocess(image2D, brightness, contrast, saturation, gamma, scale, adjustOrder)

        stackImage = np.concatenate((image2D, transformedImage), axis=1)
        plt.imshow(stackImage)
        plt.show(block=True)
