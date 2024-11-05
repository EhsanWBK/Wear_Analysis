''' ## Camera System

Provides 2 code routines for camera connection and operations for either the Baumer Camera, or 
a webcam interface. The Baumer routine is copied from the Baumer Camera documentation.

Accessible functions:
- cameraSetup()
- videoStream()
- reformatImage()

Accessible classes:
- Baumer()
- VideoCamera()
'''

import cv2
from sys import exit as ex
from os import getcwd
from os.path import join
from datetime import datetime
# from header import cameraConnection # PROBLEM WITH UPDATED STATUS MAYBE
from base64 import b64decode, b64encode
from numpy import uint8, frombuffer
from time import sleep

cameraConnection = False

try:
    import neoapi
    import vax_io

    class VideoCamera(object):
        def __init__(self):
            # setup 
            vax_io.out1.period=1000000
            vax_io.out1.duty_cycle = 500000
            vax_io.out1.enable = True

            self.camera = neoapi.Cam()
            self.camera.Connect(vax_io._som.camport)

            if self.camera.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
                self.camera.f.PixelFormat.SetString('BGR8')
                print('BGR8')
            elif self.camera.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
                self.camera.f.PixelFormat.SetString('Mono8')
                print('Mono8')

        def stopCam(self):
            self.camera.Disconnect(vax_io._som.camport)

        def startTrigger(self):
            self.camera.f.TriggerMode.value = neoapi.TriggerMode_On
            vax_io.cam_trigger.value = False
            self.camera.f.LineSelector.value = neoapi.LineSelector_Line1
            self.camera.f.LineMode.value = neoapi.LineMode_Input
            self.camera.f.TriggerSource.value = neoapi.TriggerSource_Line1

        def stopTrigger(self):
            self.camera.f.TriggerMode.value = neoapi.TriggerMode_Off

        def checkTrigger(self):
            triggerImg = self.camera.GetImage().GetNPArray()
            if triggerImg.shape == (0,0,1):
                print('Listeing to Trigger: ', triggerImg.shape)
                return False, None
            else: 
                print(triggerImg.shape)
                self.stopTrigger()
                return True, triggerImg

        def getImage(self) -> bytes:  
            self.img = self.camera.GetImage().GetNPArray()
            print(self.img.shape)
            return self.img
        
        def save_frame (self):
            print('Saving the image')
            saving = True
            if saving:
                now = datetime.now()
                filename = 'Aufnahmen/' + now.strftime('%Y-%m-%d_%H-%M-%S') + '_Aufnahme.jpg'
                return(filename, self.img) 

except:   
    class VideoCamera(object):
        def __init__(self, index: int):
            self.video = cv2.VideoCapture(index)

        def __del__(self):
            try:
                self.video.release()
            except:
                print('Cannot destroy camera')

        def getImage(self) -> bytes:
            success, image = self.video.read()
            jpeg = b64decode(image)
            jpegNp = frombuffer(jpeg, dtype=uint8)
            self.img = cv2.imdecode(jpegNp, flags=1)
            return self.img
        
        def save_frame (self):
            s, image = self.video.read()        
            if s:
                now = datetime.now()
                filename = 'Aufnahmen/' + now.strftime('%Y-%m-%d_%H-%M-%S') + '_Aufnahme.jpg'
                cv2.imwrite(filename, image)            
                return(filename, image)

def cameraSetup():
    global cameraConnection
    ''' Set-up the camera connection. If no connection can be established the function returns False.'''
    cam = None
    if not cameraConnection:
        cam = VideoCamera()
        cameraConnection = True
        return cam
    else:
        print('No Camera Connection')
        cameraConnection = False
        del cam
        return False
    
def videoStream(camera = VideoCamera):
    global cameraConnection
    ''' Create a video stream to the camera. If the connection is interupted, the video stream stops.'''
    print('Generation Video Stream')
    while cameraConnection:
        frame = camera.get_frame()
        yield frame
    print('Stopped Video Stream')

    