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
from base64 import b64decode
from numpy import uint8, frombuffer
from time import sleep

cameraConnection = False

try:
    import neoapi
    import vax_io
    class VideoCamera(object):
        def __init__(self):
            self.video_path = join(getcwd(),'temp','video_temp.avi')
            self.result = 0
            isColor = True

            # setup trigger
            vax_io.out1.period=1000000
            vax_io.out1.duty_cycle = 500000
            vax_io.out1.enable = True

            # create video; RGB or grayscaled (depends on camera -> current Baumer Camera uses RGB)
            try:
                self.camera=neoapi.Cam()
                self.camera.Connect(vax_io._som.camport)
                if self.camera.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
                    self.camera.f.PixelFormat.SetString('BGR8')
                    print('BGR8')
                elif self.camera.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
                    self.camera.f.PixelFormat.SetString('Mono8')
                    isColor = False
                    print('Mono8')
                else: print('No supported pixel format')

                # setup camera parameters          
                self.camera.f.ExposureTime.Set(10000)
                self.camera.f.AcquisitionFrameRateEnable.value = True
                self.camera.f.AcquisitionFrameRate.value = 10
                self.video=cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'XVID'), 10,
                                    (self.camera.f.Width.value, self.camera.f.Height.value), isColor)
                print('Video Created')
            except (neoapi.NeoException, Exception) as exc:
                print('error', exc)
                self.result=1    
            
        def start_cam(self):
            print('Starting Camera')
            for cnt in range(0,200):
                self.img = self.camera.GetImage().GetNPArray()
                title = 'press ESC to exit ..'
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.imshow(title, self.img)
                self.video.write(self.img)
                if cv2.waitKey(1) == 27: break

        def startTrigger(self):
            self.camera.f.TriggerMode.value = neoapi.TriggerMode_On
            vax_io.cam_trigger.value = False
            self.camera.f.LineSelector.value = neoapi.LineSelector_Line1
            self.camera.f.LineMode.value = neoapi.LineMode_Input
            self.camera.f.TriggerSource.value = neoapi.TriggerSource_Line1

        def checkTrigger(self):
            if self.cam.GetImage().GetNPArray().shape == (0,0,1):
                return False, None
            else: return True, self.cam.GetImage().GetNPArray()
            
        def __del__(self):
            self.video.release()
            cv2.destroyAllWindows()
            ex(self.result)
            print('System exit')

        def getImage(self) -> bytes:  
            image = self.camera.GetImage().GetNPArray()
            jpeg = b64decode(image)
            jpegNp = cv2.frombuffer(jpeg, dtype=uint8)
            self.img = cv2.imdecode(jpegNp, flags=1)
            return self.img
        
        def save_frame (self):
            print('Saving the image')
            saving = True
            if saving:
                now = datetime.now()
                filename = 'Aufnahmen/' + now.strftime('%Y-%m-%d_%H-%M-%S') + '_Aufnahme.jpg'
                cv2.imwrite(filename, self.image)
            
                return(filename, self.image) 

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

    