from colabcode import ColabCode
import nest_asyncio
import uvicorn
from pyngrok import ngrok
import threading
from util import url_methods

class ColabCodeExtend(ColabCode):
    def __init__(self, port=10000, password=None, authtoken=None, mount_drive=False, code=True, lab=False):
        super().__init__(port=port, password=password, authtoken=authtoken, mount_drive=mount_drive, code=code, lab=lab)

    def _start_server(self):
        if self.authtoken:
            ngrok.set_auth_token(self.authtoken)
        active_tunnels = ngrok.get_tunnels()
        for tunnel in active_tunnels:
            public_url = tunnel.public_url
            ngrok.disconnect(public_url)
        self.url = ngrok.connect(addr=self.port, bind_tls=True)
        if self._code:
            print(f"Code Server can be accessed on: {self.url}")
        else:
            print(f"Public URL: {self.url}")

    def run_app(self, app, workers=1, log_level=None):
        self.app = app
        app.servers.append(self)
        self.workers = workers
        self.log_level = log_level
        self._start()
    
    def restart(self):
        self._start()
 
    def _start(self):
        self._start_server()
        nest_asyncio.apply()
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, workers=self.workers, log_level=self.log_level)

def startServer(app, sharingDict=None, port=10000, code=False, authtoken=None, log_level=None):
    if sharingDict is not None:
        @app.on_event("startup")
        def initStartEvent():
            sharingDict["ServerURL"] = app.servers[0].url.public_url    
            sharingDict["ServerStartEvent"].set()

    server = ColabCodeExtend(port=port, code=code, authtoken=authtoken)
    server.run_app(app=app, log_level=log_level)

def autoRestartServerProcess(serverConstructingFunction, printFunction, autoCheckStatusInterval):
    def CheckServerStatus(sharingDict, interval):
        sharingDict["ServerStartEvent"].wait()
        
        isServerDown = False
        responseOutput = None
        try:
            (urlExist, responseOutput) = url_methods.urlExists(sharingDict["ServerURL"], 200, True)
            if (urlExist is False):
                isServerDown = True
        except:
                isServerDown = True

        if (isServerDown):
            if (responseOutput is not None):
                printFunction("[INFO] Server is down {}: {}".format(responseOutput.status_code, 
                    responseOutput.content))
            else:
                printFunction("[INFO] Server is down")
            sharingDict["ServerResetEvent"].set()
            return

        th = threading.Timer(interval, CheckServerStatus, args=(sharingDict, interval))
        th.daemon = True
        th.start()

    from multiprocessing import Process, Manager

    manager = Manager()
    sharingDict = manager.dict({
        "ServerURL": None,
        "ServerStartEvent": manager.Event(),
        "ServerResetEvent": manager.Event()
    })
    
    sharingDict["ServerResetEvent"].set()
    while True:
        if (sharingDict["ServerResetEvent"].wait()):
            sharingDict["ServerResetEvent"].clear()
            sharingDict["ServerStartEvent"].clear()
            sharingDict["ServerURL"] = None

            printFunction("[INFO] Starting server...")
            serverProcess = Process(target=serverConstructingFunction, args=(sharingDict,))
            serverProcess.daemon = True
            serverProcess.start()

        printFunction("[INFO] Waiting for initializing server...")
        sharingDict["ServerStartEvent"].wait()

        printFunction("[INFO] Server is running...")
        CheckServerStatus(sharingDict, autoCheckStatusInterval)

        sharingDict["ServerResetEvent"].wait()
        printFunction("[INFO] Server is down, restarting server...")
        serverProcess.terminate()
