import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from configure import config
import scipy.io as sio
import numpy as np
from util.mesh_info import MeshInfo
from skimage.transform import SimilarityTransform
from preprocessing.image3D_transforming import Image3DTransforming

class Image3DExtracting:
    # load bfm 
    bfm = None
    i3t = Image3DTransforming()

    def preprocess(self, mat):
        info = mat
        if isinstance(info, str):
            info = sio.loadmat(info)

        if ('vertices' in info):
            vertices = info['vertices']
            colors = info['colors']
            triangles = info['full_triangles']
            colors = colors/np.max(colors)
            kptIdxs = None
            if ('kptIdxs' in info):
                kptIdxs = info['kptIdxs'].squeeze()
        else:
            if (self.bfm is None):
                # load bfm 
                from model.morphable_model import MorphabelModel            
                self.bfm = MorphabelModel(config.BFM_PATH)

            pose_para = info['Pose_Para'].T.astype(np.float32)
            shape_para = info['Shape_Para'].astype(np.float32)
            exp_para = info['Exp_Para'].astype(np.float32)
            tex_para = info['Tex_Para'].astype(np.float32)

            # generate shape
            bfm_vertices = self.bfm.generate_vertices(shape_para, exp_para)
            # transform mesh
            s = pose_para[-1, 0]
            angles = pose_para[:3, 0]
            t = pose_para[3:6, 0]
            transformed_vertices = self.bfm.transform_3ddfa(bfm_vertices, s, angles, t)

            vertices = transformed_vertices.copy()
            triangles = self.bfm.full_triangles.copy()

            # generate colors
            colors = self.bfm.generate_colors(tex_para)
            colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

            # get key points
            kptIdxs = self.bfm.kpt_ind

        meshInfo = MeshInfo(vertices, colors, triangles, kptIdxs)
        if "tform" in info.keys():
            if (len(info["tform"].shape) == 2):
                tform = SimilarityTransform(matrix=info["tform"])
            else:
                matrixes = [np.array(tform) for tform in info["tform"]]
                similarityTforms = [SimilarityTransform(matrix=matrix) for matrix in matrixes]
                tform = similarityTforms[0]
                for similarityTform in similarityTforms[1:]:
                    tform = tform + similarityTform
            (h, w) = info["image_size"][0]
            meshInfo = self.i3t.preprocessTform(meshInfo, tform, h, w, translateZ=False)

        return meshInfo

if __name__ == "__main__":
    i3e = Image3DExtracting()


    matPath = r"D:\Pictures\test\image00060.mat"
    meshInfo = i3e.preprocess(matPath)
    print("aaa")

    from util import mesh_display
    mesh_display.displayPointCloudColor(meshInfo)