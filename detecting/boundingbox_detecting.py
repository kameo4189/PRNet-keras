import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from configure import config
import numpy as np

class BoundingboxDetecting:

    def detect(self, points, h, w, randomPertube=True, expandRatio=config.ORG_BB_EXTENDED_RATIO, normalBB=False):
        vertices = points.copy()

        # flip vertices along y-axis when 3D.
        if points.shape[-1] == 3:
            vertices[:,1] = h - vertices[:,1] - 1

        # get bounding box from points
        bbInfo = self.getBoundingBoxInfo(vertices, expandRatio=expandRatio, normalBB=normalBB)

        if normalBB is False and randomPertube is True:
            bbInfo = self.randomizePertubeBoundingBox(bbInfo)

        return bbInfo

    def getBoundingBoxInfo(self, points, expandRatio=config.ORG_BB_EXTENDED_RATIO, normalBB=False):
        left = np.min(points[:, 0])
        right = np.max(points[:, 0])
        top = np.min(points[:, 1])
        bottom = np.max(points[:, 1])
        if normalBB:
            return (left, right, top, bottom)
        else:
            expandCenter = np.array([right - (right - left) / 2.0, 
                bottom - (bottom - top) / 2.0])
            old_size = (right - left + bottom - top)/2
            expandSize = int(old_size*1.5)
        return (expandCenter, expandSize)

    def randomizePertubeBoundingBox(self, bbInfo, marginRatio=config.BB_MARGIN_RATIO, 
                                    marginRadomRange=config.BB_MARGIN_RANDOM_RANGE, 
                                    sizeRandomRange=config.BB_SIZE_RANDOM_RANGE):
        (center, size) = bbInfo
        newCenter = center.copy()

        margin = marginRatio * size
        t_x = margin * np.random.uniform(marginRadomRange[0], marginRadomRange[1])
        t_y = margin * np.random.uniform(marginRadomRange[0], marginRadomRange[1])
        newCenter[0] = center[0] + t_x
        newCenter[1] = center[1] + t_y

        newSize = size * np.random.uniform(sizeRandomRange[0], sizeRandomRange[1])

        return (newCenter, newSize)