''' '''

from time import sleep
from opcua.server.server import Server 
from opcua.ua import ObjectIds, DataValue, Variant, VariantType
import cv2
import base64
from sys import exit
import threading as th
try: 
    import neoapi  
    import vax_io   
except:
    print('Cannot import API.')


# Server setup
ENDPOINT = "opc.tcp://141.3.142.81:12345" # CHANGE???
# ENDPOINT = 'opc.tcp://127.0.0.1:12345'
NAMESPACE = 'CameraSpace'

# Camera setup
CAM_PIXEL_FORMAT = 'Mono8'
IS_COLOR = True

stopEvent = th.Event()
triggerSet = th.Event()

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

        self.triggerObj = self.objects.add_object(idx, "Trigger Node")
        self.triggerImgNode = self.triggerObj.add_variable(idx, 'triggerImg', Variant(False,VariantType.Boolean))
        self.triggerSetNode = self.triggerObj.add_variable(idx, 'triggerSet', Variant(False,VariantType.Boolean))
        self.triggerImgNode.set_writable()
        self.triggerSetNode.set_writable()

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
        
    def checkTrigger(self):
        trigger = self.triggerSetNode.get_value()
        if trigger: print('Trigger Val: ', trigger)
        return trigger

    def dummyStream(self, nr):
        self.dummyNr.set_value(nr)

    def getDummy(self):
        sleep(1)
        return self.dummyNr.get_value()

    def streamImg(self, imgString, trigger):
        self.imgNode.set_value(DataValue(Variant(imgString, VariantType.String)))
        if trigger: self.triggerImgNode.set_value(DataValue(Variant(trigger),VariantType.Boolean))

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
        self.camera.f.TriggerMode.value = neoapi.TriggerMode_Off
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
    
    def triggerModeOn(self):
        self.camera.f.TriggerMode.value = neoapi.TriggerMode_On
        vax_io.cam_trigger.value = False
        self.camera.f.LineSelector.value = neoapi.LineSelector_Line1
        self.camera.f.LineMode.value = neoapi.LineMode_Input
        self.camera.f.TriggerSource.value = neoapi.TriggerSource_Line1

    def triggerModeOff(self):
        self.camera.f.TriggerMode.value = neoapi.TriggerMode_Off

    def checkTrigger(self):
        triggerImg = self.camera.GetImage().GetNPArray()
        if triggerImg.shape == (0,0,1):
            return False, None
        else: 
            print(triggerImg.shape)
            ret, jpeg = cv2.imencode('.jpg', triggerImg)
            jpegString = base64.b64encode(jpeg).decode('utf-8')
            self.triggerModeOff()
            return True, jpegString
        
def castImage():
    # stream image
    # if trigger mode is set: turn off camera mode, turn on trigger mode
    # if trigger mode is unset: turn on camera mode, turn off trigger mode
    # return image string OR no image
    cam = Baumer()
    camServer = CamServer()
    serverOnline = camServer.startServer()
    triggerSignal = camServer.checkTrigger()
    print('Trigger Signal: ', triggerSignal)
    try:
        while True:
            while not triggerSignal:
                imgString = cam.getFrame()
                camServer.streamImg(imgString=imgString, trigger=False)
                triggerSignal = camServer.checkTrigger()
            cam.triggerModeOn()
            while triggerSignal:
                triggerSet, imgString = cam.checkTrigger()
                if triggerSet: 
                    print('Streaming Image')
                    camServer.streamImg(imgString=imgString, trigger=True)
                while triggerSet: 
                    print('Trigger Set: ', triggerSet)
                    triggerSet, imgString = cam.checkTrigger()
                triggerSignal = camServer.checkTrigger()
            cam.triggerModeOff()
    finally:
        serverOnline = camServer.stopServer()

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

def dummyServer():
    from os import getcwd
    from os.path import join
    from cv2 import imread
    try:
        camServer = CamServer()
        camServer.startServer()
        print('Start Server')
        dummyPath = join(getcwd(),'dummyImg.jpg') # path to dummy image
        dummyImage = imread(dummyPath, 0)
        dummyPath2 = join(getcwd(),'dummyImg2.jpg') # path to dummy image
        dummyImage2 = imread(dummyPath2, 0)

        ret, jpeg = cv2.imencode('.jpg', dummyImage)
        jpegString = base64.b64encode(jpeg).decode('utf-8')
        ret2, jpeg2 = cv2.imencode('.jpg', dummyImage2)
        jpegString2 = base64.b64encode(jpeg2).decode('utf-8')
        while True:
            camServer.streamImg(jpegString)
            sleep(1)
            camServer.streamImg(jpegString2)
            sleep(1)
    finally:
        serverOnline = camServer.stopServer()

# hostServer()
# dummyServer()

if __name__ == '__main__':
    # camThread = th.Thread(target=castImage, args=(stopEvent, triggerSet))
    # checkTrigger = th.Thread(target=checkTrigger, args=(stopEvent, triggerSet))
    castImage()
    