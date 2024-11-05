from time import sleep
from base64 import b64decode
from numpy import uint8, frombuffer
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

    def startCam(self):
        if self.cam.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
            self.cam.f.PixelFormat.SetString('BGR8')
            print('BGR8')
        elif self.cam.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
            self.cam.f.PixelFormat.SetString('Mono8')
            isColor = False
            print('Mono8')
        # self.video=cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'XVID'), 10,
        #                             (self.cam.f.Width.value, self.cam.f.Height.value), isColor)
        for cnt in range(0,200):
                self.img = self.cam.GetImage().GetNPArray()
                # title = 'press ESC to exit ..'
                # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                # cv2.imshow(title, self.img)
                # self.video.write(self.img)
                # if cv2.waitKey(1) == 27: break
                print(self.img.shape)
                sleep(0.5)


    def startTrigger(self):
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
        while True:
            # print('Trigger not set')
            sleep(1)
            self.displayImage()
        vax_io.cam_trigger.value = False
        # vax_io.cam_trigger.value = True
        # self.displayImage()
        # vax_io.cam_trigger.value = False
        # self.displayImage()
        # sleep(0.2)
        # vax_io.cam_trigger.value = True
        # self.displayImage()
        # vax_io.cam_trigger.value = False
        # self.displayImage()
        # videoStop.set()
    
    def stream(self):
        #try:
        while not videoStop.is_set():
            image = self.cam.GetImage().GetNPArray()
            jpeg = b64decode(image)
            jpegNp = frombuffer(jpeg, dtype=uint8)
            self.img = cv2.imdecode(jpegNp, flags=1)
        # except: print('ERROR STREAMING THE VIDEO')
        # finally: return True

    def displayImage(self):
        self.img = self.cam.GetImage().GetNPArray()
        print(self.img.shape)
        # print(self.img)
        return self.img#

    def stopTrigger(self):
        self.cam.f.TriggerMode.value = neoapi.TriggerMode_Off


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

# test3()

videoStop = th.Event()
baumer = Camera()

class DummyObj():
    def __init__(self) -> None:
        pass

    def printA(self):
        baumer.startCam()
        for i in range(30) and not videoStop.is_set():
            sleep(1)
            img = baumer.displayImage()          

    def printB(self):
        sleep(3)
        baumer.startTrigger()
        for i in range(2):
            sleep(1)
            print(i)
        print('Stopping Trigger')
        baumer.stopTrigger()
        # baumer.displayImage()
        # baumer.startCam()
        print('Interrupting Thread')
        videoStop.set()
        


dummy = DummyObj()
aThread = th.Thread(target=dummy.printA)
bThread = th.Thread(target=dummy.printB)

aThread.start()
bThread.start()

bThread.join()
videoStop.set()
aThread.join()
