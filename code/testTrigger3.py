import neoapi
import vax_io
from datetime import datetime
import threading as th
from time import sleep

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
            print('Listeing to Tirgger: ', triggerImg.shape)
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
        
videoStop = th.Event()
baumer = VideoCamera()

class DummyObj():
    def __init__(self) -> None:
        pass

    def printA(self):
        for i in range(10):# and not videoStop.is_set():
            sleep(1)
            img = baumer.getImage()          

    def printB(self):
        sleep(3)
        triggerSet = False
        baumer.startTrigger()
        while not triggerSet:
            sleep(1)
            triggerSet, img = baumer.checkTrigger()
        print('Stopping Trigger')
        baumer.stopTrigger()
        print('Interrupting Thread')
        videoStop.set()
        


dummy = DummyObj()
aThread = th.Thread(target=dummy.printA)
bThread = th.Thread(target=dummy.printB)

aThread.start()
bThread.start()

bThread.join()
aThread.join()