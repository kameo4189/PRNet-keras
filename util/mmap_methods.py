import os
import mmap
import sys
from util import file_methods
import win32event
import win32security

# constant values
STATUS_SIZE_BYTES = 4
RAW_FILE_MAX_SIZE_BYTES = 10 * 1024 * 1024 # 10MB
RAW_RESULT_FILE_MAX_SIZE_BYTES = 20 * 1024 * 1024 # 20MB
MESSAGE_SIZE_BYTES = 1024
STATUS_PROCESSING = 0
STATUS_OK = 1
STATUS_NG = -1

# paths for output log
STDOUT_PATH = os.sep.join([os.path.dirname(os.path.realpath(__file__)), 'stdout_analysis.txt'])
STDERR_PATH = os.sep.join([os.path.dirname(os.path.realpath(__file__)), 'stderr_analysis.txt'])

# constant win32 event values
EVENT_WAIT_INFINITE = win32event.INFINITE
WAIT_TIME_OUT = 30000 # 30s

# constant name
MMAP_SERVER_STATUS_NAME = "Session\\SERVER_STATUS"
MMAP_ANALYSIS_STATUS_NAME = "Session\\ANALYSIS_STATUS"
MMAP_IMAGE_NAME = "Session\\IMAGE"
MMAP_MAT_NAME = "Session\\MAT"
MMAP_RESULT_MAT_NAME = "Session\\RESULT_MAT"
MMAP_ERROR_MESSAGE_NAME = "Session\\ERROR_MESSAGE"
EVENT_START_ANALYSIS_NAME = "Session\\START_ANALYSIS_EVENT"
EVENT_END_ANALYSIS_NAME = "Session\\END_ANALYSIS_EVENT"

def readInt(mmapObject):
    intValue = int.from_bytes(mmapObject.read(4), byteorder=sys.byteorder)
    return intValue

def readData(mmapObject):
    size = int.from_bytes(mmapObject.read(4), byteorder=sys.byteorder)
    rawBytes = mmapObject.read(size)
    return rawBytes

def readImageData(mmapObject):
    mmapObject.seek(0)
    rawBytes = readData(mmapObject)
    image = file_methods.readImage(rawBytes, asFloat=False)
    return image

def readMatData(mmapObject):
    mmapObject.seek(0)
    rawBytes = readData(mmapObject)
    mat = file_methods.readMat(rawBytes)
    return mat

def readString(mmapObject):
    mmapObject.seek(0)
    rawBytes = readData(mmapObject)
    stringValue = rawBytes.decode()
    return stringValue

def writeInt(mmapObject, intValue):
    mmapObject.write(intValue.to_bytes(4, byteorder=sys.byteorder, signed=True))
    mmapObject.flush()

def writeIntOnly(mmapObject, intValue):
    mmapObject.seek(0)
    writeInt(mmapObject, intValue)
    mmapObject.flush()

def writeString(mmapObject, stringValue):
    mmapObject.seek(0)
    mmapObject.write(stringValue.encode())
    mmapObject.flush()

def writeData(mmapObject, bytesData):
    if (bytesData != None):
        size = len(bytesData)
        writeInt(mmapObject, size)
        mmapObject.write(bytesData)
        mmapObject.flush()

def writeDataList(mmapObject, bytesDataList):
    mmapObject.seek(0)
    if (bytesDataList != None):
        lenList = len(bytesDataList)
        writeInt(mmapObject, lenList)

        sizes = [len(bytesData) for bytesData in bytesDataList]
        for size in sizes:
            writeInt(mmapObject, size)

        for bytesData in bytesDataList:
            mmapObject.write(bytesData)
    mmapObject.flush()

def readDataList(mmapObject):
    lenList = readInt(mmapObject)
    sizes = [readInt(mmapObject) for _ in range(lenList)]
    bytesDataList = [mmapObject.read(size) for size in sizes]
    return bytesDataList

def readMatDataList(mmapObject):
    mmapObject.seek(0)
    byteDataList = readDataList(mmapObject)
    matList = [file_methods.readMat(byteData) for byteData in byteDataList]
    return matList

def createMMap(name, size, access=mmap.ACCESS_WRITE, acl=None):
    mmapObject = mmap.mmap(-1, size, name, access)
    if acl is not None:
        win32security.SetNamedSecurityInfo(name, win32security.SE_KERNEL_OBJECT, 
            win32security.DACL_SECURITY_INFORMATION, None, None, acl, None)
    return mmapObject