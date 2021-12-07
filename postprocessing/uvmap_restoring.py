import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
from configure import config
from util.mesh_info import MeshInfo

class UVMapRestoring:
    def postprocess(self, image, uvmap, tform, uvHeight=config.UV_HEIGHT, uvWidth=config.UV_WIDTH):
        (w, h) = (image.shape[1], image.shape[0])

        all_vertices = uvmap.copy()
        all_vertices = all_vertices * max(uvHeight, uvWidth) * config.UV_POS_SCALE
        if tform is not None:
            all_vertices = self.unWrap(all_vertices, tform, uvHeight, uvWidth)
        
        # 3D vertices
        vertices = self.get_vertices(all_vertices)

        # kpt indexes
        kpt_idxs = self.get_landmark_idxs(uvHeight, uvWidth)

        # corresponding colors
        colors = self.get_colors(image, vertices)

        # flip vertices along y-axis.
        vertices[:,1] = h - vertices[:,1] - 1
        
        return MeshInfo(vertices, colors, config.FACE_TRIANGLES, kpt_idxs)

    def unWrap(self, uvmap, tform, uvHeight=config.UV_HEIGHT, uvWidth=config.UV_WIDTH):
        '''
        Args:
            uvmap: the 3D uv map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(uvmap, [-1, 3]).T
        z = all_vertices[2,:] / np.mean(tform.scale)
        all_vertices[2,:] = 1
        all_vertices = np.dot(np.linalg.inv(tform.params), all_vertices)
        all_vertices = np.vstack((all_vertices[:2,:], z))
        all_vertices = np.reshape(all_vertices.T, [uvHeight, uvWidth, 3])
        return all_vertices

    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[config.UV_KPT_INDEXES[1,:], config.UV_KPT_INDEXES[0,:], :]
        return kpt

    def get_landmark_idxs(self, uvHeight=config.UV_HEIGHT, uvWidth=config.UV_WIDTH):
        uv_kpt_idxs = config.UV_KPT_INDEXES[1,:] * uvWidth + config.UV_KPT_INDEXES[0,:]
        kpt_idxs = np.nonzero(uv_kpt_idxs[:, None] == config.FACE_INDEXES)[1]

        return kpt_idxs

    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [config.UV_WIDTH*config.UV_HEIGHT, -1])
        vertices = all_vertices[config.FACE_INDEXES, :]
            
        return vertices

    def get_colors(self, image, vertices):        
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w] = image.shape[:2]
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3
        if colors.max() > 1.0:
            colors = colors / 255.

        return colors

if __name__ == "__main__":
    import numpy as np
    import scipy.io as sio
    from skimage import io
    import skimage
    import matplotlib.pyplot as plt
    from preprocessing.image3D_extracting import Image3DExtracting
    from preprocessing.image3D_to_2D import Image3DTo2D
    from preprocessing.image3D_transforming import Image3DTransforming
    from preprocessing.image2D_warping import Image2DWarping
    from detecting.boundingbox_detecting import BoundingboxDetecting
    from preprocessing.uvmap_creating import UVMapCreating
    from util import mesh_display

    # init propressors
    i3e = Image3DExtracting()
    i3t = Image3DTransforming()
    uvc = UVMapCreating()
    i3t2 = Image3DTo2D()
    bd = BoundingboxDetecting()
    i2w = Image2DWarping()
    ur = UVMapRestoring()
    

    # test for 3dfa
    image_path = r'data/IBUG_image_008_1_0.jpg'
    mat_path = r'data/IBUG_image_008_1_0.mat'

    image2D = io.imread(image_path)/255.
    [h, w, c] = image2D.shape

    meshInfo = i3e.preprocess(mat_path)    
    
    # imageFace = i3t2.preprocess(meshInfo, h, w, False)
    # imageFaceKpts = i3t2.drawKeypoints(imageFace, meshInfo)

    kpts = meshInfo.vertices[meshInfo.kptIdxs, :]
    bbInfo = bd.detect(kpts, h, w, True)

    (wrappedImage, tform) = i2w.preprocess(image2D, bbInfo, 
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    wrappedMesh = i3t.preprocessTform(meshInfo, tform, h, w, 
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    (uv_position_map, uv_texture_map) = uvc.preprocess(wrappedMesh)

    wrappedImageubyte = skimage.img_as_ubyte(wrappedImage)
    uv_position_map = uv_position_map.astype(np.float32)

    # wrappedImageFace = i3t2.preprocess(wrappedMesh,
    #     config.IMAGE_WIDTH, config.IMAGE_HEIGHT, False)
    # wrappedImageFaceKpts = i3t2.drawKeypoints(wrappedImageFace, wrappedMesh)
    
    # uv_texture_map_kpts = uvc.drawKeypoints(uv_texture_map, wrappedMesh)
    # uv_position_map_vis = (uv_position_map)/max(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    # uv_position_map_kpts = uvc.drawKeypoints(uv_position_map_vis, wrappedMesh)

    uv_position_map_ref = np.zeros([config.IMAGE_HEIGHT * config.IMAGE_WIDTH, 3])
    uv_position_map_ref[config.FACE_INDEXES, :] = np.reshape(uv_position_map, [-1, 3])[config.FACE_INDEXES, :]
    uv_position_map_ref = np.reshape(uv_position_map_ref, 
        [config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3])
    uv_position_map_ref_vis = (uv_position_map_ref)/max(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    plt.rcParams["figure.figsize"] = (20,20)
    plt.subplot(2,3,1)
    plt.imshow(wrappedImageubyte)
    # plt.imshow(wrappedImageFaceKpts, alpha=.7)
    # plt.subplot(2,3,2)
    # plt.imshow(uv_texture_map)
    # plt.imshow(uv_texture_map_kpts, alpha=.7)
    # plt.subplot(2,3,3)
    # plt.imshow(uv_position_map_vis)
    # plt.imshow(uv_position_map_kpts, alpha=.7)
    plt.subplot(2,3,4)
    plt.imshow(uv_position_map_ref_vis)
    plt.show()

    uvr = UVMapRestoring()
    restoredMeshInfoNoneWrap = uvr.postprocess(wrappedImage, uv_position_map, None)
    restoredMeshInfoWrap = uvr.postprocess(image2D, uv_position_map, tform)

    # angles = [-50, -30, -20, 0, 20, 30, 50]
    # restoredMeshInfoNoneWraps = [i3t.preprocess(restoredMeshInfoNoneWrap, -1, [0, a, 0], [0, 0, 0]) for a in angles]
    # restoredImageFaceNoneWraps = [i3t2.preprocess(m, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, True) for m in restoredMeshInfoNoneWraps]

    # for i, image in enumerate(restoredImageFaceNoneWraps):
    #     plt.subplot(2,4,i+1)
    #     plt.imshow(image)
    # plt.show()
   
    # restoredImageFaceNoneWrapKpts = i3t2.drawKeypoints(wrappedImage, restoredMeshInfoNoneWrap)
    restoredImageFaceWrap = i3t2.preprocess(restoredMeshInfoWrap, h, w, False)
    restoredImageFaceWrapKpts = i3t2.drawKeypoints(image2D, restoredMeshInfoWrap)

    mesh_display.displayPointCloudColor(restoredMeshInfoNoneWrap)
    mesh_display.displayPointCloudColor(restoredMeshInfoWrap)
    mesh_display.displayMesh(restoredMeshInfoNoneWrap)
    mesh_display.displayMesh(restoredMeshInfoWrap)

    # plt.subplot(2,3,1)
    # plt.imshow(wrappedImage)
    # plt.imshow(restoredImageFaceNoneWrapKpts, alpha=.5)
    # plt.subplot(2,3,2)
    # plt.imshow(image2D)
    # plt.imshow(restoredImageFaceWrapKpts, alpha=.5)
    # plt.show()