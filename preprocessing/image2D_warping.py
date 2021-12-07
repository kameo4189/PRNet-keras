import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
import skimage.transform

class Image2DWarping:

    def preprocess(self, image, bbInfo, outHeight, outWidth):
        (center, size) = bbInfo

        # crop and record the transform parameters
        src_pts = np.array([[center[0] - size/2, center[1] - size/2], 
            [center[0] - size/2, center[1] + size/2], 
            [center[0] + size/2, center[1] - size/2]])
        DST_PTS = np.array([[0, 0], [0, outHeight - 1], [outWidth - 1, 0]])
        tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
        warpedImage = skimage.transform.warp(image, tform.inverse, output_shape=(outHeight, outWidth))

        return (warpedImage, tform)

