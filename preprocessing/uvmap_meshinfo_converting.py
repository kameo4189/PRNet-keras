import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
from configure import config
from util.mesh_info import MeshInfo

class UVMapMeshInfoConverting:
    def preprocess(self, image, uvmap, tform,
        uvHeight=config.UV_HEIGHT, uvWidth=config.UV_WIDTH):
        (w, h) = (image.shape[1], image.shape[0])

        all_vertices = uvmap
        if isinstance(uvmap, str):
           all_vertices = np.load(uvmap)
      
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
        
        return MeshInfo(vertices, colors, None, kpt_idxs)

    def unWrap(self, uvmap, tform, uvHeight=config.UV_HEIGHT, uvWidth=config.UV_WIDTH):
        '''
        Args:
            uvmap: the 3D uv map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(uvmap, [-1, 3]).T
        z = all_vertices[2,:] / tform.scale
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
        return uv_kpt_idxs

    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        vertices = np.reshape(pos, [config.UV_WIDTH*config.UV_HEIGHT, -1])
            
        return vertices

    def get_colors(self, image, vertices):        
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3
        if colors.max() > 1.0:
            colors = colors / 255.

        return colors