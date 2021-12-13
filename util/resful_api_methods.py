import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from pydantic import BaseModel
from typing import List
from typing import Optional
from configure import config_evaluating

class ImageMatData(BaseModel):
    rawImage: str
    rawMats: List[str]

class InputData(BaseModel):
    matType: Optional[str] = 'normal'
    imageMatList: List[ImageMatData]

class FaceMeshData(BaseModel):
    rawMesh: str
    faceBB: List[int]

class ImageMeshData(BaseModel):
    faceMeshList: List[FaceMeshData] = []

class OutputData(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        if ("numImages" in data.keys()):
            numImages = data["numImages"]
            self.imageMeshList = [ImageMeshData() for _ in range(numImages)]
       
    imageMeshList: List[ImageMeshData] = []

class ImageMatEvalData(BaseModel):
    rawImage: str
    rawOrgMats: List[str]
    rawPredictMats: List[str]

class InputEvalData(BaseModel):
    imageMatList: List[ImageMatEvalData]

class MeshEvalResult(BaseModel):
    evalValues: List[float] = [0 for _ in range(len(config_evaluating.EVALUATION_TYPES_DISPLAY))] 

class ImageEvalResult(BaseModel):
    imageEvalValues: List[MeshEvalResult] = []

class OutputEvalData(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        if ("numImages" in data.keys()):
            numImages = data["numImages"]
            self.evaluationDataList = [ImageEvalResult() for _ in range(numImages)]   
    evaluationTypes: List[str] = config_evaluating.EVALUATION_TYPES_DISPLAY    
    evaluationDataList: List[ImageEvalResult] = []