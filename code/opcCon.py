from opcua.client.client import Client
from opcua.ua import DataValue, Variant, VariantType
from time import sleep
import base64
from numpy import frombuffer, uint8
from cv2 import imdecode


URL = 'opc.tcp://141.3.142.81:12345'
NAMESPACE = 'CameraSpace'

class CamClient:

    def __init__(self) -> None:
        print('Initializing Client')
        # try:
        self.client = Client(url=URL, timeout=100000)
        self.client.connect()
        self.client.get_namespace_array()

        self.objects = self.client.get_objects_node()
        self.nodes = self.objects.get_children()
        print('Nodes found on OPC UA Server:\n',self.nodes)
        # except:
        #     print('Error')

    def getImage(self):
        ''' Read out bytestring from image node. 
        Name of image node: "Image Node". '''
        imgNode = self.nodes[1] # FIND INDEX OF IMAGE NODE
        imgData = imgNode.get_children()[0]
        jpegString = imgData.get_value()
        jpeg = base64.b64decode(jpegString)
        jpegNp = frombuffer(jpeg, dtype=uint8)
        self.img = imdecode(jpegNp, flags=1)
        return self.img
    
    def getTrigger(self):
        ''' Read out boolean from trigger node. Default False.'''
        triggerNode = self.nodes[2]
        trigger= triggerNode.get_children()[0]
        print('Trigger Image Received: ', trigger)
        return trigger.get_value()
    
    def setTriggerMode(self, mode: bool):
        triggerNode = self.nodes[2]
        triggerMode= triggerNode.get_children()[1]
        triggerMode.set_value(DataValue(Variant(mode,VariantType.Boolean)))
        
    def receivedTrigger(self):
        triggerNode = self.nodes[2]
        triggerImg= triggerNode.get_children()[0]
        triggerImg.set_value(DataValue(Variant(False,VariantType.Boolean)))

    def stopClient(self):
        print('\t- Terminated Client.')
        self.client.disconnect()
