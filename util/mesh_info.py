import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import scipy.io as sio
import numpy as np
from io import BytesIO
from configure import config
from detecting.boundingbox_detecting import BoundingboxDetecting

class MeshInfo:
    vertices = None
    colors = None
    triangles = None
    kptIdxs = None
    _bd = BoundingboxDetecting()

    @property
    def kptPos(self):
        return self.vertices[self.kptIdxs]

    def kptPos2D(self, h):
        return h - self.vertices[self.kptIdxs][:, 1] - 1

    def bbSize(self, h, w):
        (l,r,t,b) = self._bd.detect(self.vertices, h, w, False, normalBB=True)
        bbSize = np.sqrt(np.power(r-l,2) + np.power(b-t,2)).astype("float32")
        return bbSize

    def outerInterocularDistance(self):
        if self.kptIdxs is None or len(self.kptIdxs) != 68:
            raise NotImplementedError()
        left = self.vertices[self.kptIdxs[37]]
        right = self.vertices[self.kptIdxs[46]]
        distance = np.sqrt(np.sum(np.power(left - right, 2)))
        return distance

    def __init__(self, vertices=None, colors=None, triangles=None, kptIdxs=None):
        self.vertices = vertices
        self.colors = colors
        self.triangles = triangles
        self.kptIdxs = kptIdxs
        if (self.vertices is not None and self.vertices.shape[-1] != 3):
            self.vertices = np.transpose(self.vertices, (1, 0)) if self.vertices is not None else None 
            self.colors = np.transpose(self.colors, (1, 0)) if self.colors is not None else None
            self.triangles = np.transpose(self.triangles, (1, 0)) if self.triangles is not None else None

    def copy(self):
        newMeshInfo = MeshInfo()
        newMeshInfo.vertices = self.vertices.copy() if self.vertices is not None else None
        newMeshInfo.colors = self.colors.copy() if self.colors is not None else None
        newMeshInfo.triangles = self.triangles.copy() if self.triangles is not None else None
        newMeshInfo.kptIdxs = self.kptIdxs.copy() if self.kptIdxs is not None else None
        return newMeshInfo

    def getDict(self, matType="normal", decimalDtype=None, intergerDtype=None):
        meshDict = {}
        verticesDtype = decimalDtype if decimalDtype is not None else self.vertices.dtype
        colorDtype = decimalDtype if decimalDtype is not None else self.colors.dtype

        kptPointDtype = intergerDtype if intergerDtype is not None else self.kptIdxs.dtype

        import open3d as o3d
        o3dMesh = o3d.geometry.TriangleMesh()
        o3dMesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        o3dMesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        o3dMesh.compute_vertex_normals()
        vertex_normals = np.asarray(o3dMesh.vertex_normals)
        bounding_box = np.concatenate((o3dMesh.get_min_bound(), o3dMesh.get_max_bound())).astype(verticesDtype)

        if matType == "dotnet":
            meshDict['vertices'] = np.transpose(self.vertices, (1, 0)).astype(verticesDtype)
            meshDict['colors'] = np.transpose(self.colors, (1, 0)).astype(colorDtype)
            meshDict['full_triangles'] = np.transpose(self.triangles, (1, 0)).astype(kptPointDtype)
            meshDict['vertex_normals'] = np.transpose(vertex_normals, (1, 0)).astype(verticesDtype)
        else:
            meshDict['vertices'] = self.vertices.astype(verticesDtype)
            meshDict['colors'] = self.colors.astype(colorDtype)
            meshDict['full_triangles'] = self.triangles.astype(kptPointDtype)
            meshDict['vertex_normals'] = vertex_normals.astype(verticesDtype)
        meshDict['kptIdxs'] = self.kptIdxs.astype(kptPointDtype)
        meshDict['bounding_box'] = bounding_box
        return meshDict

    def save(self, path, dtype=None, decimalDtype=None, intergerDtype=None):
        sio.savemat(path, self.getDict(dtype, decimalDtype, intergerDtype))

    def toBytes(self, matType="normal", decimalDtype=None, intergerDtype=None):
        with BytesIO() as f:
            sio.savemat(f, self.getDict(matType, decimalDtype, intergerDtype))
            return f.getvalue()

    def toCommonFaceRegionMesh(self):
        commonTriangles = config.FACE_INDEXES_TO_ORG_TRIANGLES
        commonTriWeights = config.FACE_INDEXES_TO_ORG_TRIWEIGHTS[:, :, np.newaxis]
        vertices = (commonTriWeights[:, 0] * self.vertices[commonTriangles[:, 0]] +
            commonTriWeights[:, 1] * self.vertices[commonTriangles[:, 1],:] + 
            commonTriWeights[:, 2] * self.vertices[commonTriangles[:, 2]])
        colors = None
        if self.colors is not None:
            colors = (commonTriWeights[:, 0] * self.colors[commonTriangles[:, 0]] +
                commonTriWeights[:, 1] * self.colors[commonTriangles[:, 1]] + 
                commonTriWeights[:, 2] * self.colors[commonTriangles[:, 2]])
        triangles = config.FACE_TRIANGLES

        uv_kpt_idxs = config.UV_KPT_INDEXES[1,:] * config.UV_WIDTH + config.UV_KPT_INDEXES[0,:]
        kpt_idxs = np.nonzero(uv_kpt_idxs[:, None] == config.FACE_INDEXES)[1]

        return MeshInfo(vertices, colors, triangles, kpt_idxs)

class Obj:
    colorVertices = None
    triangles = None

    def __init__(self, meshInfo):
        if meshInfo.colors is not None:
            self.colorVertices = np.column_stack((meshInfo.vertices, meshInfo.colors))
        else:
            self.colorVertices = np.column_stack((meshInfo.vertices, np.full_like(meshInfo.vertices, 1.0)))
        i = [2, 1, 0]
        self.triangles = meshInfo.triangles[:, i]

    def _saveCore(self, path, colorVertices, triangles):
        with open(path, 'w') as f:
            # write vertices & colors
            for i in range(colorVertices.shape[0]):
                s = 'v {} {} {} {} {} {}\n'.format(colorVertices[i, 0], colorVertices[i, 1], colorVertices[i, 2],
                    colorVertices[i, 3], colorVertices[i, 4], colorVertices[i, 5])
                f.write(s)

            # write f: ver ind/ uv ind
            for i in range(triangles.shape[0]):
                s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
                f.write(s)

    def save(self, path):
        self._saveCore(path, self.colorVertices, self.triangles)

    @staticmethod
    def saveFromMeshInfo(path, meshInfo):
        Obj(meshInfo).save(path)