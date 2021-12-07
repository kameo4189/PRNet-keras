import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from os import path
from configure import config
from pathlib import Path
import numpy as np
import glob
from io import BytesIO 
from skimage import io
import scipy.io as sio
from util import url_methods
import traceback
from skimage.color import rgba2rgb
import base64
import skimage

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def getAllFiles(inputFolder, exts):
    filterExts = ["*" + ext for ext in exts]

    allFilePaths = []
    for filterExt in filterExts:
        filePaths = sorted(glob.glob(os.sep.join([inputFolder, "**", filterExt]), 
            recursive=True))
        allFilePaths.extend(filePaths)

    return allFilePaths

def getAllImages(inputFolder):
    return getAllFiles(inputFolder, image_types)

def getParentPath(path):
    return str(Path(path).parent)

def getFileName(path):
    return Path(path).name

def getFileNameWithoutExt(path):
    return Path(path).stem

def getExt(path):
    return Path(path).suffix

def getFilePathWithOtherExt(filePath, ext, parentPath=None):
    if parentPath is None:
        parentPath = getParentPath(filePath)
    nameWithoutExt = getFileNameWithoutExt(filePath)
    newName = nameWithoutExt + ext
    newPath = path.sep.join([parentPath, newName])
    return newPath

def getFileNameWithOtherExt(filePath, ext):
    nameWithoutExt = getFileNameWithoutExt(filePath)
    newName = nameWithoutExt + ext
    return newName

def getFilePathWithOtherParent(filePath, parentPath, posfix=None):
    name = getFileName(filePath)
    if (posfix is not None):
        nameWithoutExt = getFileNameWithoutExt(filePath)
        ext = getExt(filePath)
        name = nameWithoutExt + "_" + posfix + ext
    newPath = path.sep.join([parentPath, name])
    return newPath

def makeDirs(dirName, parentPath):
    dirPath = os.path.sep.join([parentPath, dirName])
    if os.path.exists(dirPath) is False:
        os.makedirs(dirPath)
    return dirPath

def getImageWithMatList(inputFolder):
    allImagePaths = getAllFiles(inputFolder, image_types)

    imageWithMatList = []
    for imagePath in allImagePaths:
        mathPath = getFilePathWithOtherExt(imagePath, ".mat")

        if (path.exists(mathPath)):
            imageWithMatList.append((imagePath, mathPath))
        else:
            print("mat for image {0} doesn't exist".format(imagePath))
            imageWithMatList.append((imagePath, mathPath))

    return imageWithMatList

def getFilePathsWithSameStructure(inputPaths, srcDirPath, destDirPath, ext):
    relSrcPaths = [os.path.relpath(path, srcDirPath) for path in inputPaths]
    outputPaths = [os.path.sep.join([destDirPath, path]) for path in relSrcPaths]         
    outputPaths = [getFilePathWithOtherExt(path, ext) for path in outputPaths]
    outputPaths = [path if os.path.exists(path) else "" for path in outputPaths]
    return outputPaths

def getSubject(filePath):
    fileNameWithoutExt = getFileNameWithoutExt(filePath)
    subjectParts = fileNameWithoutExt.split("_")
    if (len(subjectParts) <= 2):
        subject = subjectParts[0]
    elif ("image" in subjectParts):
        subject = "_".join(subjectParts[:4])
    else:
        subject = "_".join(subjectParts[:3])
    return subject

def getImageWithMatWithSubjectList(inputFolder):
    allImagePaths = getAllFiles(inputFolder, image_types)

    imageWithMatWithSubjectList = []
    for imagePath in allImagePaths:
        mathPath = getFilePathWithOtherExt(imagePath, ".mat")
        subject = getSubject(imagePath)

        if (path.exists(mathPath)):
            imageWithMatWithSubjectList.append((imagePath, mathPath, subject))
        else:
            print("mat for image {0} doesn't exist".format(imagePath))

    return imageWithMatWithSubjectList

def getExtractedPath(zipDirPath, zipFileName, zipFileUrl, extractDirPath, extractDirName):
    zipFilePath = os.path.sep.join([zipDirPath, zipFileName])
    extractedFilePath = os.path.sep.join([extractDirPath, extractDirName])
    extractedToFilePath = os.path.sep.join([extractDirPath, 
        getFileNameWithoutExt(zipFileName), extractDirName])

    if (os.path.exists(zipFilePath) is False and
        os.path.exists(extractedFilePath) is False and
        os.path.exists(extractedToFilePath) is False and
        zipFileUrl is None):
        print("Not exist data paths for {}...".format(extractDirName))
        return None
    
    if (os.path.exists(extractedFilePath) is False and
        os.path.exists(extractedToFilePath) is False):
        if (zipFileUrl is None and os.path.exists(zipFilePath) is False):
            print("Not exist url for {}...".format(extractDirName))
            return None

        print("Downloading and unzip the dataset {}...".format(extractDirName))
        try:
            if (os.path.exists(zipFilePath) is False):
                url_methods.download(zipFileUrl, extractDirPath)
            url_methods.unzip(zipFilePath, extractDirPath)
        except:
            print("Occuring error when download and unzip {}".format(extractDirName))
            print(traceback.format_exc())
            return None

    resultPath = extractedFilePath if os.path.exists(extractedFilePath) is True else extractedToFilePath
    resultPath = resultPath if os.path.exists(resultPath) else None
    return resultPath

def readRawFile(path, toBytes=False):
    if os.path.exists(path) is False:
        if toBytes:
            return b''
        else:
            return None

    try:
        with open(path, "rb") as file:
            if toBytes:
                return file.read()
            else:
                return BytesIO(file.read())
    except Exception as e:
        print("File {} reading error: {}".format(path, str(e)))
        if toBytes:
            return b''
        else:
            return None

def bytesToString(bytes):
    return base64.b64encode(bytes).decode()

def stringToBytes(string):
    return base64.b64decode(string)

def readImage(image, asFloat=True):
    if isinstance(image, np.ndarray):
        if asFloat:
            return skimage.img_as_float(image)
        else:
            return skimage.img_as_ubyte(image)

    inputImage = image
    if isinstance(image, bytes):
        inputImage = BytesIO(image)
    resultImage = io.imread(inputImage)
    if (resultImage.shape[-1] == 4):
        resultImage = (rgba2rgb(resultImage) * 255).astype("uint8")

    if asFloat:
        return skimage.img_as_float(resultImage)
    else:
        return skimage.img_as_ubyte(resultImage)

def saveImage(imagePath, image):
    io.imsave(imagePath, image)

def readMat(mat):
    inputMat = mat
    if isinstance(mat, bytes):
        if len(mat) == 0:
            print("Mat file appears to be empty")
            return None
        mopt_bytes = mat[:4]
        mopt_ints = np.ndarray(shape=(4,), dtype=np.uint8, buffer=mopt_bytes)
        if 0 in mopt_ints:
            print("Mat file appears to be empty")
            return None
        inputMat = BytesIO(mat)
    try:
        return sio.loadmat(inputMat)
    except Exception as e:
        print("Mat reading error: {}".format(str(e)))
        return None

def saveMat(matPath, mat, addItems=None):
    if addItems is not None:
        mat.update(addItems)
    sio.savemat(matPath, mat)