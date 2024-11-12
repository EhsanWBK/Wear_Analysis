import neoapi; import vax_io

cameraConnection = False

class Baumer():

    def __init__(self):
        print('Starting Camera')
        self.camera = neoapi.Cam()
        self.camera.Connect()
        print('Camera connected.')
        if self.camera.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
            self.camera.f.PixelFormat.SetString('BGR8'); print('BGR8')
        elif self.camera.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
            self.camera.f.PixelFormat.SetString('Mono8'); print('Mono8')
        self.camera.f.ExposureTime.Set(10000)
        self.camera.f.AcquisitionFrameRateEnable.value = True
        self.camera.f.AcquisitionFrameRate.value = 10
        self.camera.f.TriggerMode.value = neoapi.TriggerMode_Off
        print('Video Stream established.')
        print('Setup complete.')
        self.trigger=False

    def startCam(self):
        print('Starting Stream')
        for cnt in range(0,200): self.img = self.camera.GetImage().GetNPArray()

    def stopCam(self):
        print('Disconnecting Camera.'); self.camera.Disconnect()

    def getFrame(self) -> bytes:
        return self.camera.GetImage().GetNPArray()

    def triggerModeOn(self):
        self.camera.f.TriggerMode.value = neoapi.TriggerMode_On
        vax_io.cam_trigger.value = False
        self.camera.f.LineSelector.value = neoapi.LineSelector_Line1
        self.camera.f.LineMode.value = neoapi.LineMode_Input
        self.camera.f.TriggerSource.value = neoapi.TriggerSource_Line1
        self.triggerStatus = True

    def triggerModeOff(self):
        self.camera.f.TriggerMode.value = neoapi.TriggerMode_Off
        self.trigger = False

    def triggerStatus(self):
        return self.trigger