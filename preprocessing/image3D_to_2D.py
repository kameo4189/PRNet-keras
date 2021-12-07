import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from util import mesh_render, triangle_process, uv_process, mesh_transform
from util.cython import mesh_render as render_colors_cython
from configure import config

class Image3DTo2D:
    def preprocess(self, meshInfo, h, w, translate2center=True):
        # 1. transform 3D vertices in 3D space to coresponding 3D vertices in 2D coordinates system
        projected_vertices = meshInfo.vertices.copy() # use standard camera & orth projection here

        # tranlate to center of 2d or not
        if translate2center:
            image_vertices = mesh_transform.to_image_center(projected_vertices, h, w)
        else:
            image_vertices = mesh_transform.to_image(projected_vertices, h, w)

        # 2. render 2d facial image      
        image = render_colors_cython.render_colors(image_vertices, meshInfo.triangles, meshInfo.colors, 
            h, w, config.COLOR_CHANNEL)

        return image

    def drawKeypoints(self, image, meshInfo, pointSize=2):
        return mesh_render.render_kps(image, meshInfo.vertices, meshInfo.kptIdxs, True, pointSize)