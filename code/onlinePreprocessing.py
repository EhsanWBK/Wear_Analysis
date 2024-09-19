''' ## Online Image Data Preprocessing

DOES NOT WORK AT THE MOMENT!

Processing of Image Data in the Online Model Application Process.
Either processing stack (numpy array) of images or single images.

Accessible Functions:
- preProcImage()
- processStack()
'''

from crop_align import cropImage, alignImage
from numpy import ndarray

def cropImage(image: ndarray) -> ndarray:
    ''' Crop single image. '''
    image = cropImage(imageData=image)
    return image

def alignImage(image: ndarray) -> ndarray: # ALIGNMENT FOR SINGLE IMAGE NOT POSSIBLE!
    ''' Align single image. '''
    image = alignImage(imageData=image)
    return image

def convertImage(image: ndarray) -> ndarray: # is this necessary? Needs TIF format
    ''' Convert file type of single image. '''
    return image
 

def preProcImage(imageData: ndarray, operation: str) -> ndarray:
    ''' Start pre-processing of single image. "Operation" gives order of operations. '''
    curImg = cropImage(image=imageData)
    # curImg = alignImage(image=curImg)
    return curImg

def processStack(imageData: ndarray, operation: str) -> ndarray: # IS THIS NEEDED? FUNCTIONAL IN OFFLINE PREPROCESSING
    ''' Process stack of images as numpy array. Operations need to be specified as array.
    E.g.: ['align', 'crop', 'convert]
    '''
    for images in range(len(imageData)):
        preProcImage(imageData=imageData[images], operation=operation)
    return

