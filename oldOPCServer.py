''' '''

from time import sleep
from opcua.server.server import Server 
from opcua.ua import ObjectIds, DataValue, Variant, VariantType
import cv2
import base64
from sys import exit
try: 
    import neoapi     
except:
    print('Cannot import API.')


# Server setup
ENDPOINT = "opc.tcp://141.3.142.81:12345" # CHANGE???
NAMESPACE = 'CameraSpace'

# Camera setup
CAM_PIXEL_FORMAT = 'Mono8'
IS_COLOR = True

class CamServer:
    def __init__(self) -> None:
        self.server = Server()
        self.server.set_endpoint(ENDPOINT)
        idx = self.server.register_namespace(NAMESPACE)
        print('Namespace Index: ',idx)

        self.objects = self.server.get_objects_node()
        self.imgObj = self.objects.add_object(idx, "Image Node")
        self.imgNode = self.imgObj.add_variable(idx, 'image',Variant("",VariantType.String))
        self.imgNode.set_writable()

        # dummy
        self.dummy = self.objects.add_object('ns=2;s="DUM"', "Dummy")
        self.dummyNr = self.dummy.add_variable('ns=2;s="DUM_NR"', "Dummy Nummer", 1)

    def startServer(self):
        try: self.server.start()
        except: 
            print('Failed to Start the Server')
            return False
        finally: 
            print('Server Online')
            return True
    
    def stopServer(self):
        try: self.server.stop()
        except: 
            print('Failed to Stop the Server')
            return True
        finally: 
            print('Server Offline')
            return False

    def dummyStream(self, nr):
        self.dummyNr.set_value(nr)

    def getDummy(self):
        sleep(1)
        return self.dummyNr.get_value()

    def streamImg(self, imgString):
        self.imgNode.set_value(DataValue(Variant(imgString, VariantType.String)))

class Baumer():

    def __init__(self) -> None:
        print('Starting Camera')
        self.camera = neoapi.Cam()
        self.camera.Connect()
        print('Camera connected.')
        self.camera.f.PixelFormat.SetString(CAM_PIXEL_FORMAT)
        self.camera.f.ExposureTime.Set(10000)
        self.camera.f.AcquisitionFrameRateEnable.value = True
        self.camera.f.AcquisitionFrameRate.value = 10
        print('Video Stream established.')
        print('Setup complete.')
        

    def startCam(self):
        print('Starting Stream')
        for cnt in range(0,200):
            self.img = self.camera.GetImage().GetNPArray()
            
    def getFrame(self) -> bytes:
        self.img = self.camera.GetImage().GetNPArray()
        ret, jpeg = cv2.imencode('.jpg', self.img)
        jpegString = base64.b64encode(jpeg).decode('utf-8')
        return jpegString

def hostServer():
    ''' Endless loop. Hosts server and streams image data. '''
    try:
        print('Start Server')
        dummyIterate = 0
        camServer = CamServer()
        cam = Baumer()
        serverOnline = camServer.startServer()
        if cam == None: print('Camera is None')
        while True:
            imgString = cam.getFrame() # retreive image from camera stream
            camServer.streamImg(imgString)
            # camServer.dummyStream(dummyIterate)
            # print(camServer.getDummy())
            # dummyIterate +=1
    finally:
        serverOnline = camServer.stopServer()

hostServer()