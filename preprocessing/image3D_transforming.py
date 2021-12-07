import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
from util import mesh_transform, common_methods

class Image3DTransforming:
    def preprocess(self, meshInfo, meshSize, angles, translations):
        vertices = meshInfo.vertices.copy() #- np.mean(meshInfo.vertices, 0)[np.newaxis, :]
        s = 1
        if (meshSize > 0):
            s = meshSize/(np.max(meshInfo.vertices[:,1]) - np.min(meshInfo.vertices[:,1]))
        R = common_methods.angle2matrix(angles) 
        t = translations
        vertices = mesh_transform.similarity_transform(vertices, s, R, t)

        transformedMeshInfo = meshInfo.copy()
        transformedMeshInfo.vertices = vertices

        return transformedMeshInfo

    def preprocessTform(self, meshInfo, tform, inHeight=None, inWidth=None, 
        outHeight=None, outWidth=None, translateZ=True):
        tMeshInfo = meshInfo.copy()

        # flip vertices along y-axis.
        if (inHeight is not None):
            tMeshInfo.vertices[:, 1] = inHeight - tMeshInfo.vertices[:, 1] - 1

        tMeshInfo.vertices[:, 2] = 1
        tMeshInfo.vertices = np.dot(tMeshInfo.vertices, tform.params.T)
        tMeshInfo.vertices[:, 2] = meshInfo.vertices[:, 2] * np.mean(tform.scale) # scale z
        if translateZ:
            tMeshInfo.vertices[:, 2] = tMeshInfo.vertices[:, 2] - np.min(tMeshInfo.vertices[:, 2]) # translate z

        # flip vertices along y-axis.
        if (inHeight is not None):
            if (outHeight is None):
                outHeight = inHeight        
            tMeshInfo.vertices[:,1] = outHeight - tMeshInfo.vertices[:,1] - 1

        return tMeshInfo

if __name__ == "__main__":
    from image3D_extracting import Image3DExtracting
    from preprocessing.image3D_to_2D import Image3DTo2D
    import matplotlib.pyplot as plt
    from skimage import io

    i3e = Image3DExtracting()
    i3t = Image3DTransforming()
    i3t2 = Image3DTo2D()

    # meshInfo = i3e.preprocess("data/example1.mat")
    # meshInfo = i3t.preprocess(meshInfo, 180, [0, 0, 0], [0, 0, 0])

    # # transform
    # angles = [-50, -30, -20, 0, 20, 30, 50]
    # meshInfos = [i3t.preprocess(meshInfo, -1, [0, a, 0], [0, 0, 0]) for a in angles]
    # images = [i3t2.preprocess(m, 256, 256) for m in meshInfos]

    # for i, image in enumerate(images):
    #     plt.subplot(2,4,i+1)
    #     plt.imshow(image)
    # plt.show()

    image_path = r'data/IBUG_image_008_1_0.jpg'
    mat_path = r'data/IBUG_image_008_1_0.mat'

    image2D = io.imread(image_path)/255.
    [h, w, c] = image2D.shape
    meshInfo = i3e.preprocess(mat_path)
    imageOrg = i3t2.preprocess(meshInfo, h, w, False) 

    # angles = [-50, -30, -20, 0, 20, 30, 50]
    # meshInfos = [i3t.preprocess(meshInfo, -1, [0, a, 0], [0, 0, 0]) for a in angles]
    # images = [i3t2.preprocess(m, h, w, False) for m in meshInfos]
    # images.insert(0, imageOrg)

    # for i, image in enumerate(images):
    #     plt.subplot(2,4,i+1)
    #     plt.imshow(image)
    # plt.show()

    from preprocessing.uvmap_creating import UVMapCreating
    from preprocessing.image2D_warping import Image2DWarping
    from detecting.boundingbox_detecting import BoundingboxDetecting
    from preprocessing.geometric_transforming import GeometricTransforming
    from configure import config
    from configure import config_augmenting as confa
    from numpy.random import uniform
    
    gt = GeometricTransforming()
    i2w = Image2DWarping()
    uvc = UVMapCreating()
    bd = BoundingboxDetecting()

   
    kpts = meshInfo.vertices[meshInfo.kptIdxs, :]
    bbInfo = bd.detect(kpts, h, w, True)
    (wrappedImage, tform) = i2w.preprocess(image2D, bbInfo, 
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    wrappedMesh = i3t.preprocessTform(meshInfo, tform, h, w, 
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    wrappedImageFace = i3t2.preprocess(wrappedMesh,
        config.IMAGE_WIDTH, config.IMAGE_HEIGHT, False)
    wrappedImageFaceKpts = i3t2.drawKeypoints(wrappedImageFace, wrappedMesh)

    uv_position_map = uvc.preprocess(wrappedMesh, type="pos")
    uv_position_map_vis = (uv_position_map)/max(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    uv_position_map_kpts = uvc.drawKeypoints(uv_position_map_vis, wrappedMesh)

    [h, w, c] = wrappedImageFace.shape
    (rotMin, rotMax) = confa.TRANSFORMING_ROTATION_RANGE
    (transMin, transMax) = confa.TRANSFORMING_TRANSLATION_RATIO_RANGE
    (scaleMin, scaleMax) = confa.TRANSFORMING_SCALE_RATIO_RANGE

    rotation = uniform(rotMin, rotMax)
    translationX = uniform(transMin, transMax)
    translationY = uniform(transMin, transMax)
    translations = (translationX * w, translationY * h)
    scale = uniform(scaleMin, scaleMax)
    (augImage, tform) = gt.preprocess(wrappedImageFace, None, scale, rotation, translations)
    augPosMap = i3t.preprocessTform(uv_position_map, tform, h, w)
    augPosMapVis = (uv_position_map)/max(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    augPosMapKpts = uvc.drawKeypoints(augPosMapVis, wrappedMesh)

    plt.rcParams["figure.figsize"] = (20,20)
    plt.subplot(2,3,1)
    plt.imshow(wrappedImage)
    plt.imshow(wrappedImageFaceKpts, alpha=.7)
    plt.subplot(2,3,2)
    plt.imshow(uv_position_map_vis)
    plt.imshow(uv_position_map_kpts, alpha=.7)
    plt.subplot(2,3,3)
    plt.imshow(uv_position_map_vis)
    plt.imshow(uv_position_map_kpts, alpha=.7)
    plt.show()
