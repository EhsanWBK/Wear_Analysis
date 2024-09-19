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
from header import cameraConnection # PROBLEM WITH UPDATED STATUS MAYBE
from base64 import b64encode
from numpy import ndarray

try:
    baumer = True
    import neoapi
except:
    baumer = False

class Baumer(object):
    def __init__(self):
        """
        Instructor of Camera

        Parameters
        ----------
        index : int
            Index of the camera. The first camera is always at index 0.

        Returns
        -------
        None.

        """
        self.video_path = join(getcwd(),'Temp','video_temp.avi')
        self.result = 0
        isColor = True

        # create video; RGB or grayscaled (depends on camera -> current Baumer Camera uses RGB)
        try:
            self.camera=neoapi.Cam()
            self.camera.Connect()
            if self.camera.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
                self.camera.f.PixelFormat.SetString('BGR8')
                print('BGR8')
            elif self.camera.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
                self.camera.f.PixelFormat.SetString('Mono8')
                isColor = False
                print('Mono8')
            else:
                print('No supported pixel format')
                ex(0)

            # setup camera parameters            
            self.camera.f.ExposureTime.Set(10000)
            self.camera.f.AcquisitionFrameRateEnable.value = True
            self.camera.f.AcquisitionFrameRate.value = 10
            
            # create video stream
            self.video=cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'XVID'), 10,
                                (self.camera.f.Width.value, self.camera.f.Height.value), isColor)
            print('Video Created')

        except (neoapi.NeoException, Exception) as exc:
            print('error', exc)
            self.result=1
            
        
    def start_cam(self):
        # camera setup
        print('Starting Camera')

        for cnt in range(0,200):
            self.img = self.camera.GetImage().GetNPArray()
            title = 'press ESC to exit ..'
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, self.img)
            self.video.write(self.img)
            
            if cv2.waitKey(1) == 27:
                break
          
    def __del__(self):
        """
        Destructur of camera.
        Releases the connection between the Python Kernel and the camera.

        Returns
        -------
        None.

        """
        self.video.release()
        cv2.destroyAllWindows()
        ex(self.result)
        print('System exit')


    def get_frame(self) -> bytes:
        """
        Sends captured frames.

        Returns
        -------
        bytes
            Python bytes (encoded jpg image).

        """
        if baumer:
            startPoint = (100 ,int(self.camera.f.Height.value-150))
            endPoint = (100+576, int(self.camera.f.Height.value-100))

            # capture image and transform it to numpy array
            self.image = self.camera.GetImage().GetNPArray()

            # add overlay to image
            overlay = self.image.copy()
            cv2.rectangle(overlay, startPoint, endPoint, (0,0,0),-1)
            cv2.addWeighted(overlay, 1, self.image, 0, 0, self.image)
        else:
            success, image = self.video.read()
            # We are using Motion JPEG, but OpenCV defaults to capture raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def save_frame (self):
        """
    
        Returns
        -------
        None.

        If reading video frame was successful, save current frame in Aufnahmen.

        Returns
        -------
        list
            DESCRIPTION.

        """
        print('Saving the image')
        saving = True
        if saving:
            now = datetime.now()
            filename = 'Aufnahmen/' + now.strftime('%Y-%m-%d_%H-%M-%S') + '_Aufnahme.jpg'
            cv2.imwrite(filename, self.image)
           
            return(filename, self.image)  
        
class VideoCamera(object):
    '''
    Created on Mon Oct 17 10:43:19 2022

    @author: Michael Poncelet
    '''
    def __init__(self, index: int):
        """
        Instructor of Camera

        Parameters
        ----------
        index : int
            Index of the camera. The first camera is always at index 0.

        Returns
        -------
        None.

        """
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(index)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        

    def __del__(self):
        """
        Destructur of camera.
        Releases the connection between the Python Kernel and the camera.

        Returns
        -------
        None.

        """
        try:
            self.video.release()
        except:
            print('Cannot destroy camera')

    def get_frame(self) -> bytes:
        """
        Sends captured frames.

        Returns
        -------
        bytes
            Python bytes (encoded jpg image).

        """
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def save_frame (self):
        """
        

        Returns
        -------
        None.

        """
        """
        If reading video frame was successful, save current frame in Aufnahmen.

        Returns
        -------
        list
            DESCRIPTION.

        """
        s, image = self.video.read()        
        if s:
            #image = segmentation.resize_image(image)
            now = datetime.now()
            filename = 'Aufnahmen/' + now.strftime('%Y-%m-%d_%H-%M-%S') + '_Aufnahme.jpg'
            cv2.imwrite(filename, image)
            # ret, jpeg = cv2.imencode('.jpg', image)
            # return [filename, jpeg.tobytes()]
           
            return(filename, image)

def cameraSetup():
    ''' Set-up the camera connection. If no connection can be established the function returns False.'''
    if not cameraConnection:
        try: cam = Baumer()
        except: cam = VideoCamera()
        return cam
    else:
        print('No Camera Connection')
        cameraConnection = False
        del cam
        return False
    
def videoStream(camera = VideoCamera):
    ''' Create a video stream to the camera. If the connection is interupted, the video stream stops.'''
    print('Generation Video Stream')
    while cameraConnection:
        frame = camera.get_frame()
        yield frame
    print('Stopped Video Stream')

def reformatImage(image: ndarray):
    ''' Reformats image and converts to utf-8 BLOB (binary large object).'''
    # print(image)
    jpeg = cv2.imencode('.jpeg', image)[1]
    byte_data = jpeg.tobytes() # deconstruct into byte
    blob = b64encode(byte_data)
    # print(blob)
    return blob.decode("utf-8")
    