import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import skimage.transform as skt
import math
from preprocessing.image3D_transforming import Image3DTransforming
import numpy as np

class GeometricTransforming:
    i3t = Image3DTransforming()

    def preprocess(self, image, meshInfo, scale, rotation, translation, mode="reflect"):
        (width, height) = (image.shape[1], image.shape[0])

        shift_y, shift_x = np.array(image.shape[:2]) / 2.
        tf_rotate = skt.SimilarityTransform(rotation=np.deg2rad(rotation))
        tf_shift = skt.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = skt.SimilarityTransform(translation=[shift_x, shift_y])
        tformRotate = (tf_shift + (tf_rotate + tf_shift_inv))
        tformScaleTranslate = skt.SimilarityTransform(scale=(scale, scale), translation=translation)        
        tform = tformRotate + tformScaleTranslate
        warpedImage = skt.warp(image, tform.inverse, mode=mode)

        if meshInfo is not None:
            warpedMeshInfo = self.i3t.preprocessTform(meshInfo, tform, height, width, translateZ=False)
            return (warpedImage, warpedMeshInfo, tform)

        return (warpedImage, tform)

if __name__ == "__main__":
    import numpy as np
    import scipy.io as sio
    from skimage import io
    import matplotlib.pyplot as plt
    from preprocessing.image3D_extracting import Image3DExtracting
    from preprocessing.image3D_to_2D import Image3DTo2D

    gt = GeometricTransforming()
    i3e = Image3DExtracting()
    i3t2 = Image3DTo2D()

    image_path = r"K:\Study\CaoHoc\LuanVan\dataset\AFLW2000\image00013.jpg"
    mat_path = r"K:\Study\CaoHoc\LuanVan\dataset\AFLW2000\image00013.mat"

    image2D = io.imread(image_path)/255.
    [h, w, c] = image2D.shape

    meshInfo = i3e.preprocess(mat_path)
    # imageFace = i3t2.preprocess(meshInfo, h, w, False)
    # imageFaceKpts = i3t2.drawKeypoints(imageFace, meshInfo)

    scale = 1.3
    rotation = 30
    translation = (0.1 * w, 0.05 * h)
    (transformedImage, transformedMeshInfo, tform) = gt.preprocess(image2D, meshInfo, scale, rotation, translation, True)
    transformedImageFace = i3t2.preprocess(transformedMeshInfo, h, w, False)
    transformedImageFaceKpts = i3t2.drawKeypoints(transformedImageFace, transformedMeshInfo)

    stackImage = np.concatenate((image2D, transformedImage, transformedImageFaceKpts), axis=1)
    plt.imshow(stackImage)
    # plt.imshow(transformedImage)
    # plt.imshow(transformedImageFaceKpts, alpha=.5)
    plt.show(block=True)