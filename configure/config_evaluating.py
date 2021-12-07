import os

EVALUATION_TYPES = ["FACE_ALIGNMENT_SPARSE_2D", "FACE_ALIGNMENT_SPARSE_3D",
    "FACE_ALIGNMENT_DENSE_2D", "FACE_ALIGNMENT_DENSE_3D",
    "FACE_RECONSTRUCTION_2D", "FACE_RECONSTRUCTION_3D"]
EVALUATION_TYPES_DISPLAY = ["sparse 2D", "sparse 3D", "dense 2D", "dense 3D",
    "reconstruction 2D", "reconstruction 3D"]
EVALUATION_LINE_STYLES = ["b-", "b--", "g-", "g--", "y-", "y--"]
EVALUATION_FIGNAME_CED = "FACE_RESTORING_EVALUATION_CED.png"
EVALUATION_FORMAT_FIGNAME_CED = "FACE_RESTORING_EVALUATION_CED_{}.png"
EVALUATION_STATUS_JSON_NAME = "FaseRestoringEvaluation.json"

class EVALUATION_INFO:
    __FILENAME_NME_POSFIX = "_NME.csv"
    __FILENAME_CED_POSFIX = "_CED.csv"
    __FIGNAME_CED_POSFIX = "_CED.png"

    def __init__(self, typeName, parentPath):
        self.TYPE_NAME = typeName
        self.TYPE_DISPLAY_NAME = EVALUATION_TYPES_DISPLAY[EVALUATION_TYPES.index(typeName)]
        self.FILEPATH_NME = os.path.sep.join([parentPath, typeName + self.__FILENAME_NME_POSFIX])
        self.FILEPATH_CED = os.path.sep.join([parentPath, typeName + self.__FILENAME_CED_POSFIX])
        self.FIGPATH_CED = os.path.sep.join([parentPath, typeName + self.__FIGNAME_CED_POSFIX])
        self.LINE_STYLE = EVALUATION_LINE_STYLES[EVALUATION_TYPES.index(typeName)]

# define the path to the output directory used for storing plots,
# classification reports, etc.
BASE_EVALUATION_PATH = r"evaluation"
def createEvaluationInfos(baseEvaluationPath=BASE_EVALUATION_PATH):
    global EVALUATION_INFOS
    global EVALUATION_PATH
    global EVALUATION_FIGPATH_CED
    global EVALUATION_FORMAT_FIGPATH_CED
    global EVALUATION_STATUS_JSON_PATH

    basePath = baseEvaluationPath
    if baseEvaluationPath is None:
        basePath = ""

    EVALUATION_PATH = basePath
    if os.path.exists(EVALUATION_PATH) is False:
        os.makedirs(EVALUATION_PATH)

    EVALUATION_INFOS = {}
    for type in EVALUATION_TYPES:
        EVALUATION_INFOS[type] = EVALUATION_INFO(type, EVALUATION_PATH)

    EVALUATION_FIGPATH_CED = os.path.sep.join([EVALUATION_PATH, EVALUATION_FIGNAME_CED])
    EVALUATION_FORMAT_FIGPATH_CED = os.path.sep.join([EVALUATION_PATH, EVALUATION_FORMAT_FIGNAME_CED])
    EVALUATION_STATUS_JSON_PATH = os.path.sep.join([EVALUATION_PATH, EVALUATION_STATUS_JSON_NAME])