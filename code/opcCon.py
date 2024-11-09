from opcua.client.client import Client
from opcua.ua import DataValue, Variant, VariantType
import base64
from numpy import frombuffer, uint8
from cv2 import imdecode

# URL = 'opc.tcp://141.3.142.81:12345'
URL = 'opc.tcp://127.0.0.1:12345'
NAMESPACE = 'CameraSpace'

class CamClient:
    def __init__(self) -> None:
        print('Initializing Client')
        self.client = Client(url=URL, timeout=100000)
        self.client.connect()
        self.client.get_namespace_array()
        self.objects = self.client.get_objects_node()
        self.nodes = self.objects.get_children()
        self.imgNode = self.nodes[1]
        self.triggerNode = self.nodes[2]
        print('Nodes found on OPC UA Server:\n',self.nodes)

    def getImage(self):
        ''' Read out bytestring from image node.Name of image node: "Image Node". '''
        imgData = self.imgNode.get_children()[0]
        jpegString = imgData.get_value()
        jpeg = base64.b64decode(jpegString)
        jpegNp = frombuffer(jpeg, dtype=uint8)
        return imdecode(jpegNp, flags=1)
    
    def getTrigger(self):
        trigger= self.triggerNode.get_children()[0]
        return trigger.get_value()
    
    def setTriggerMode(self, mode: bool):
        triggerMode= self.triggerNode.get_children()[1]
        triggerMode.set_value(DataValue(Variant(mode,VariantType.Boolean)))
        
    def receivedTrigger(self):
        triggerImg= self.triggerNode.get_children()[0]
        triggerImg.set_value(DataValue(Variant(False,VariantType.Boolean)))

    def stopClient(self):
        print('\t- Terminated Client.'); self.client.disconnect()
