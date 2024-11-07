from time import sleep
from base64 import b64decode
from numpy import uint8
import cv2
import threading as th
try:
    import neoapi
    import vax_io
except:
    print('ERROR IMPORTING LIBRARIES')

videoStop = th.Event()

class Camera(object):
    def __init__(self) -> None:
        # enable pwm output out1
        # implicit sets internal camera line output to exposureActive
        vax_io.out1.period = 1000000
        vax_io.out1.duty_cycle = 500000
        vax_io.out1.enable = True

        # connect internal camera and set exposure time
        self.cam = neoapi.Cam()
        self.cam.Connect(vax_io._som.camport)
        self.cam.f.TriggerMode.value = neoapi.TriggerMode_On
        self.cam.f.ExposureTime.value = 100000

    def softwareTrigger(self):
        self.cam.f.TriggerSource.value = neoapi.TriggerSource_Software
        self.displayImage()
        self.cam.f.TriggerSoftware.Execute()
        self.displayImage()
        sleep(0.2)
        self.cam.f.TriggerSoftware.Execute()
        self.displayImage()

    def hardwareTrigger(self):
        vax_io.cam_trigger.value = False
        self.cam.f.LineSelector.value = neoapi.LineSelector_Line1
        self.cam.f.LineMode.value = neoapi.LineMode_Input
        self.cam.f.TriggerSource.value = neoapi.TriggerSource_Line1
        self.displayImage()
        vax_io.cam_trigger.value = True
        self.displayImage()
        vax_io.cam_trigger.value = False
        self.displayImage()
        sleep(0.2)
        vax_io.cam_trigger.value = True
        self.displayImage()
        vax_io.cam_trigger.value = False
        self.displayImage()
        videoStop.set()
    
    def stream(self):
        try:
            while not videoStop.is_set():
                image = self.camera.GetImage().GetNPArray()
                jpeg = b64decode(image)
                jpegNp = cv2.frombuffer(jpeg, dtype=uint8)
                self.img = cv2.imdecode(jpegNp, flags=1)
        except: print('ERROR STREAMING THE VIDEO')
        finally: return True

    def displayImage(self):
        self.img = self.camera.GetImage().GetNPArray()
        print(self.img.shape)
        print(self.img)
        return self.img

def test1():
    ''' Test Setup of the Camera. '''
    baumer = Camera()
    img = baumer.displayImage()
    return True

def test2():
    ''' Test Software Trigger. '''
    baumer = Camera()
    baumer.softwareTrigger()
    return True

def test3():
    ''' Test Physical Trigger. '''
    baumer = Camera()
    baumer.hardwareTrigger()
    return True

def test4():
    ''' Test Video Stream'''
    baumer = Camera()
    baumer.stream()
    return

def test5():
    ''' Test Video Stream and Physical Trigger. '''
    # threading setup
    
    baumer = Camera()
    streamThread = th.Thread(baumer.stream)
    triggerThread = th.Thread(baumer.hardwareTrigger)

    streamThread.start()
    triggerThread.start()

    streamThread.join()
    triggerThread.join()


# videoStop = th.Event()


# class DummyObj():
#     def __init__(self) -> None:
#         pass

#     def printA(self):
#         while not videoStop.is_set():
#             sleep(1)
#             print('a')

#     def printB(self):
#         for i in range(10):
#             sleep(1)
#             print('b')
#         videoStop.set()
#         print('Interrupting Thread')


# dummy = DummyObj()
# aThread = th.Thread(target=dummy.printA)
# bThread = th.Thread(target=dummy.printB)

# aThread.start()
# bThread.start()

# aThread.join()
# bThread.join()