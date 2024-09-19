''' ## Functions for Image Segmentation

Accessible Functions:
- displayPred()
- segmentImages()
- onlineSegmentation()
'''

from numpy import expand_dims, uint8, ndarray, array
import matplotlib.pyplot as plt
from random import randint
from header import currentModel
from generalUtensils import loadCurModel
from onlinePreprocessing import processStack, preProcImage
from sklearn.preprocessing import normalize


def displayPred(testImg: ndarray, groundTruth: ndarray, pred) -> None:
    plt.figure(figsize=(16,8))
    plt.subplot(131) # plot original image
    plt.title('Testing Image')
    plt.imshow(testImg[0], cmap='gray')
    plt.subplot(132) # plot true mask of image
    # plt.title('Testing Label')
    # plt.imshow(groundTruth[:,:,0], cmap='gray')
    # plt.subplot(133) # plot predicted mask
    plt.title('Predicted Mask')
    plt.imshow(pred, cmap='gray')
    plt.show()

def segmentImages(testData: dict, randomize: bool = True) -> None:
    ''' Testing model on random images. Takes in test data and randomization command.
    Prints out images (pyplot).
    If "randomize" = True: pick random image from stack.
    '''
    model = currentModel if currentModel else loadCurModel() # load model if not existing already
    if randomize: # maybe change to: pick random or take in single
        idx = randint(0, len(testData['xTest'])) # create random index
        targetImg = testData['xTest'][idx] # select random image
        groundTruth = testData['yTest'][idx]
    else:
        idx = 0
    targetImgNorm = targetImg[:,:,0][:,:,None] 
    targetImgInput = expand_dims(targetImgNorm, 0) # expand dimension
    pred = (model.predict(targetImgInput)[0,:,:,0] > 0.2).astype(uint8)
    displayPred(testImg=targetImg, groundTruth=groundTruth, pred=pred) # displays results


def onlineSegmentation(image: ndarray, model) -> ndarray:
    ''' Online Segmentation of images. Takes in single image and model to segment with.
    Returns predicted mask of the input image.
    
    Image needs preprocessing! '''
    # image preprocessing
    print(image.shape)
    # mask = model.predict(image)
    # print(mask)
    image = expand_dims(normalize(array(image[0,:,:,0]), axis=1),2)
    image = image[:,:,0][:,:,None]
    image = expand_dims(image,0)  
    print(image.shape)
    mask = (model.predict(image)[0,:,:,0]>0.2).astype(uint8)
    return mask