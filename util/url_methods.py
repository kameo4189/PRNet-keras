import os
from os import path
import requests
import re
from tqdm import tqdm
import zipfile

def getFilename_fromCd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]

def download(url, outputDir):
    fileName = os.path.basename(url)
    downloadFilePath = os.path.sep.join([outputDir, fileName])
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(downloadFilePath, 'wb') as f:
            pbar = tqdm(desc="Downloading {}".format(fileName), total=int(r.headers['Content-Length']))
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

def unzip(filePath, outputDir):
    fileName = os.path.basename(filePath)
    with zipfile.ZipFile(filePath, "r") as zf:
        for member in tqdm(zf.infolist(), desc='Extracting {}'.format(fileName)):
            try:
                zf.extract(member, outputDir)
            except:
                pass

def urlExists(url, status_code_ok=None, returnResponse=False):
    result = False
    responseOutput = None

    if (url is None):
        result = False
    try:
        responseOutput = requests.get(url)
        if (status_code_ok is not None):
            result = (responseOutput.status_code == status_code_ok)
        else:
            result = (responseOutput.status_code < 400)
    except:
        result = False

    if (returnResponse):
        return (result, responseOutput)
    return result