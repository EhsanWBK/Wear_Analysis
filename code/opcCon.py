from opcua.client.client import Client
from time import sleep
import base64
from numpy import frombuffer, uint8
from cv2 import imdecode


URL = 'opc.tcp://127.0.0.1:12345'
NAMESPACE = 'CameraSpace'

class CamClient:

    def __init__(self) -> None:
        print('Initializing Client')
        try:
            self.client = Client(url=URL, timeout=100000)
            self.client.connect()
            self.client.get_namespace_array()

            self.objects = self.client.get_objects_node()
            self.nodes = self.objects.get_children()
            print('Nodes found on OPC UA Server:\n',self.nodes)
        except:
            print('Error')

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
    
    def getDummy(self):
        ''' '''

        dummyNode = self.nodes[2]
        dummy= dummyNode.get_children()[0]
        print(dummy.get_value())

    def stopClient(self):
        print('\t- Terminated Client.')
        self.client.disconnect()
