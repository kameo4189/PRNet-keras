# import nessesary packages
import os
from configure import config_logging
from fastapi import FastAPI
from message_logging.server_logging import MessageLog, FileLog 
import tqdm
from util import file_methods
from util import dweet_methods
from util import server_methods
from util import datetime_methods
import threading

outputLogFile = open(config_logging.LOG_OUTPUT_FILE_PATH, 'a')
def out(message):
    print(message)
    outputLogFile.write(message+"\n")
    outputLogFile.flush()

def ConstructLoggerServer(sharingDict):
    app = FastAPI()

    def postServerUrl():
        try:
            urlThingName = config_logging.DWEET_LOGGING_URL_THING_NAME
            urlPayload = config_logging.DWEET_LOGGING_URL_PAYLOAD
            urlPayloadKey = config_logging.DWEET_LOGGING_URL_PAYLOAD_KEY
            urlPayload[urlPayloadKey] = app.servers[0].url.public_url
            response = dweet_methods.dweet_for(urlThingName, urlPayload)
            out("[INFO] Post server url succesfully: {}".format(response))
        except Exception as e:
            out("[Error] Error occurred when post server url: {}".format(e))
        th = threading.Timer(3600.0, postServerUrl)
        th.daemon = True
        th.start()

    @app.on_event("startup")
    def initialize():
        # try:
        #     postServerUrl()
        #     sharingDict["ServerURL"] = app.servers[0].url.public_url
        # finally:
        #     sharingDict["ServerStartEvent"].set()
        postServerUrl()

    @app.get('/')
    def index():
        return {'message': 'This is the homepage of the API '}

    @app.get('/test')
    def test():
        sharingDict["ServerURL"] = ""

    @app.post('/log_message')
    def logMessage(data: MessageLog):
        source = data.source
        message = data.message
        out("[{}][{}] {}".format(datetime_methods.nowStr(), source, message))

    @app.post('/log_file')
    def logFile(data: FileLog):
        source = data.source
        name = data.name
        byteString = data.content
        bytes = file_methods.stringToBytes(byteString)
        filePath = os.path.sep.join([config_logging.LOG_OUTPUT_PATH, name])
        out("[{}][{}] Writing received file to path {}...".format(datetime_methods.nowStr(), source, filePath))
        open(filePath, "wb").write(bytes)

    server_methods.startServer(app, sharingDict=sharingDict, port=10000, code=False, authtoken=None, log_level="error")

if __name__ == "__main__":
    server_methods.autoRestartServerProcess(ConstructLoggerServer, out, 3*60)