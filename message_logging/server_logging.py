import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from configure import config_logging
from pydantic import BaseModel
import requests
import threading
from util import file_methods
from util import dweet_methods
from util import url_methods

class MessageLog(BaseModel):
    source: str
    message: str

class FileLog(BaseModel):
    source: str
    name: str
    content: str

class ServerLogger(object):
    messageLogURLPath = "/log_message"
    fileLogURLPath = "/log_file"
    serverURL = ""

    def __init__(self, logSource=""):
        self.terminal = sys.stdout
        self.logSource = logSource

    def write(self, message):
        if message != '\n':
            self.terminal.write(message + '\n')
            thr = threading.Thread(target=self.send, args=(message, self.logSource,), kwargs={})
            thr.start()

    def log(self, message):
        if message != '\n':
            thr = threading.Thread(target=self.send, args=(message,self.logSource,), kwargs={})
            thr.start()

    def flush(self):
        pass

    @staticmethod
    def getLogServerUrl():
        if (url_methods.urlExists(ServerLogger.serverURL)):
            return ServerLogger.serverURL
        
        sys.stdout.terminal.write("[WARNING] Logging server url not exists, try to get new one\n")  
        currentURL = ServerLogger.serverURL
        try:
            thingName = config_logging.DWEET_LOGGING_URL_THING_NAME
            key = config_logging.DWEET_LOGGING_URL_PAYLOAD_KEY
            response = dweet_methods.get_latest_dweet_for(thingName)
            ServerLogger.serverURL = response[0]["content"][key]
        except Exception as e:
            sys.stdout.terminal.write("[ERROR] Get logging server url exception: {}\n".format(str(e)))      
            ServerLogger.serverURL = currentURL
        return ServerLogger.serverURL

    @staticmethod
    def send(message, logSource):
        try:
            serverURL = ServerLogger.getLogServerUrl()

            if serverURL != "":
                if (os.path.exists(message)):
                    ServerLogger.sendFile(message, serverURL, logSource)
                else:
                    ServerLogger.sendMessage(message, serverURL, logSource)
            else:
                sys.stdout.terminal.write("[ERROR] Log server url not exist")
        except Exception as e:
            sys.stdout.terminal.write("[ERROR] Send log exception: " + str(e))      

    @staticmethod
    def sendMessage(msg, url, logSource):
        messageUrl = url + ServerLogger.messageLogURLPath
        msgData = MessageLog(source=logSource, message=msg)
        try:
            response = requests.post(messageUrl, json=msgData.dict())
            if response.status_code != 200:
                sys.stdout.terminal.write("[ERROR] Post message log failed {}: {}\n".format(response.status_code, response.content))
        except Exception as e:
            sys.stdout.terminal.write("[ERROR] Post message log exception: {}\n".format(str(e)))

    @staticmethod
    def sendFile(filePath, url, logSource):
        messageUrl = url + ServerLogger.fileLogURLPath
        bytes = file_methods.readRawFile(filePath, True)
        byteString = file_methods.bytesToString(bytes)
        fileName = file_methods.getFileName(filePath)
        msgData = FileLog(source=logSource, name=fileName, content=byteString)
        try:
            response = requests.post(messageUrl, json=msgData.dict())
            if response.status_code != 200:
                sys.stdout.terminal.write("[ERROR] Post file log error failed {}: {}".format(response.status_code, response.content))
        except Exception as e:
            sys.stdout.terminal.write("[ERROR] Post file log exception: {}\n".format(str(e)))