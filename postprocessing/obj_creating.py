import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np
from configure import config
from util.mesh_info import Obj
from postprocessing.uvmap_restoring import UVMapRestoring

class ObjCreating:
    uvr = UVMapRestoring()

    def postprocess(self, path, meshInfo):
        Obj.saveFromMeshInfo(path, meshInfo)
       
    def postprocessFromPosMap(self, path, image, uvmap, tform=None,
        uvHeight=config.UV_HEIGHT, uvWidth=config.UV_WIDTH):
        meshInfo = self.uvr.postprocess(image, uvmap, tform, uvHeight, uvWidth)
        Obj.saveFromMeshInfo(path, meshInfo)