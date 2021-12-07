import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from util import mesh_render, uv_process, mesh_transform
from util.cython import mesh_render as mesh_render_cython
from configure import config
import numpy as np

class UVMapCreating:
    imageHeight = config.IMAGE_HEIGHT
    imageWidth = config.IMAGE_WIDTH
    colorChannel = config.COLOR_CHANNEL
    uvHeight = config.UV_HEIGHT
    uvWidth = config.UV_WIDTH
    renderColorsFunct = staticmethod(mesh_render_cython.render_colors)

    def changeRenderMode(self, mode="cython"):
        UVMapCreating.renderColorsFunct = staticmethod(mesh_render_cython.render_colors)
        if (mode == "numba"):
            UVMapCreating.renderColorsFunct = staticmethod(mesh_render.render_colors)

    def preprocess(self, meshInfo, translate2center=False, type="all", h=None, w=None):
        imageHeight = self.imageHeight if h is None else h
        imageWidth = self.imageWidth if w is None else w
        if meshInfo.vertices.size == self.uvHeight*self.uvWidth*3:
            uv_position_map = meshInfo.vertices.copy()
            # flip vertices along y-axis.
            uv_position_map[:,1] = imageHeight - uv_position_map[:,1] - 1
            uv_position_map = uv_position_map.reshape([self.uvHeight, self.uvWidth, 3])
            return uv_position_map

        # 1. initialize uv coordinates
        uv_coords = uv_process.load_uv_coords(config.BFM_UV_PATH)
        uv_coords = uv_process.process_uv(uv_coords, self.uvHeight, self.uvWidth)

        uv_position_map = None
        if (type == "all" or type == "pos"):
            # 2. transform 3D vertices in 3D space to coresponding 3D vertices in 2D coordinates system
            projected_vertices = meshInfo.vertices.copy() # use standard camera & orth projection here
            # tranlate to center of 2d or not
            if translate2center:
                image_vertices = mesh_transform.to_image_center(projected_vertices, imageHeight, 
                    imageWidth)
            else:
                image_vertices = mesh_transform.to_image(projected_vertices, imageHeight, 
                    imageWidth)

            # 3. render position map on uv coordinates
            position = image_vertices.copy()
            position[:,2] = position[:,2] - np.min(position[:,2])
            uv_position_map = UVMapCreating.renderColorsFunct(uv_coords, meshInfo.triangles, position, 
                self.uvHeight, self.uvWidth, self.colorChannel)

        # 4. render colors map on uv coordinates
        uv_texture_map = None
        if (meshInfo.colors is not None and (type == "all" or type == "tex")):
            uv_texture_map = UVMapCreating.renderColorsFunct(uv_coords, meshInfo.triangles, meshInfo.colors, 
                self.uvHeight, self.uvWidth, self.colorChannel)

        if (type == "pos"):
            return uv_position_map
        if (type == "tex"):
            return uv_texture_map
        return (uv_position_map, uv_texture_map)

    def drawKeypoints(self, image, meshInfo, pointSize=2):
        # 1. initialize uv coordinates
        uv_coords = uv_process.load_uv_coords(config.BFM_UV_PATH)
        uv_coords = uv_process.process_uv(uv_coords, self.uvHeight, self.uvWidth)

        return mesh_render.render_kps(image, uv_coords, meshInfo.kptIdxs, False, pointSize)

if __name__ == "__main__":
    import numpy as np
    import scipy.io as sio
    from skimage import io
    import matplotlib.pyplot as plt
    from preprocessing.image3D_extracting import Image3DExtracting
    from preprocessing.image3D_to_2D import Image3DTo2D
    from preprocessing.image3D_transforming import Image3DTransforming
    from preprocessing.image2D_warping import Image2DWarping
    from detecting.boundingbox_detecting import BoundingboxDetecting
    from postprocessing.uvmap_restoring import UVMapRestoring

    # init propressors
    i3e = Image3DExtracting()
    i3t = Image3DTransforming()
    uvc = UVMapCreating()
    i3t2 = Image3DTo2D()
    bd = BoundingboxDetecting()
    i2w = Image2DWarping()
    ur = UVMapRestoring()

    # # load data     
    meshInfo = i3e.preprocess("data/example1.mat")

    # # transform    
    # meshInfo = i3t.preprocess(meshInfo, config.MESH_SIZE, [0, 0, 0], [0, 0, 0])

    # import time
    # start = time.time()
    # for i in range(10):
    #     (uv_position_map, uv_texture_map) = uvc.preprocess(meshInfo)
    # end = time.time()
    # print(end - start)

    # uvc.changeRenderMode("numba")
    # start = time.time()
    # for i in range(10):
    #     (uv_position_map, uv_texture_map) = uvc.preprocess(meshInfo)
    # end = time.time()
    # print(end - start)
    
    # exit()
    
    # image = i2c.preprocess(meshInfo.vertices, meshInfo.triangles, meshInfo.colors)

    # plt.subplot(1,3,1)
    # plt.imshow(image)
    # plt.subplot(1,3,2)
    # plt.imshow(uv_texture_map)
    # plt.subplot(1,3,3)
    # plt.imshow((uv_position_map)/max(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    # plt.show()

    # test for 3dfa
    image_path = r'data/IBUG_image_008_1_0.jpg'
    mat_path = r'data/IBUG_image_008_1_0.mat'

    image2D = io.imread(image_path)/255.
    [h, w, c] = image2D.shape

    meshInfo = i3e.preprocess(mat_path)
    imageFace = i3t2.preprocess(meshInfo, h, w, False)
    imageFaceKpts = i3t2.drawKeypoints(imageFace, meshInfo)

    kpts = meshInfo.vertices[meshInfo.kptIdxs, :]
    bbInfo = bd.detect(kpts, h, w, True)
    bbInfo2 = bd.detect(meshInfo.vertices, h, w, True)

    (wrappedImage, tform) = i2w.preprocess(image2D, bbInfo, 
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    wrappedMesh = i3t.preprocessTform(meshInfo, tform, h, w, 
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    wrappedImageFace = i3t2.preprocess(wrappedMesh,
        config.IMAGE_WIDTH, config.IMAGE_HEIGHT, False)
    wrappedImageFaceKpts = i3t2.drawKeypoints(wrappedImageFace, wrappedMesh)

    (uv_position_map, uv_texture_map) = uvc.preprocess(wrappedMesh)
    uv_position_map_vis = (uv_position_map)/max(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    uv_position_map_kpts = uvc.drawKeypoints(uv_position_map_vis, wrappedMesh)
    uv_texture_map_kpts = uvc.drawKeypoints(uv_texture_map, wrappedMesh)

    tempPosMap = uv_position_map.copy()
    tempPosMap = np.reshape(tempPosMap, [-1, 3])
    face_uv = np.zeros_like(tempPosMap)
    face_uv[config.FACE_INDEXES, :] = tempPosMap[config.FACE_INDEXES, :]
    face_uv = np.reshape(face_uv, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3])
    face_uv_vis = (face_uv)/max(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    plt.rcParams["figure.figsize"] = (20,20)
    plt.subplot(2,3,1)
    plt.imshow(image2D)
    plt.imshow(imageFaceKpts, alpha=.7)
    plt.subplot(2,3,2)
    plt.imshow(wrappedImage)
    plt.imshow(wrappedImageFaceKpts, alpha=.7)
    plt.subplot(2,3,3)
    plt.imshow(uv_texture_map)
    plt.imshow(uv_texture_map_kpts, alpha=.7)
    plt.subplot(2,3,4)
    plt.imshow(uv_position_map_vis)
    plt.imshow(uv_position_map_kpts, alpha=.7)
    plt.subplot(2,3,5)
    plt.imshow(face_uv_vis)
    plt.scatter(x=config.UV_KPT_INDEXES[0,:], y=config.UV_KPT_INDEXES[1,:], s=2)
    plt.show()


