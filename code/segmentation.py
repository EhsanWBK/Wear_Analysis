''' ## Functions for Image Segmentation

Accessible Functions:
- displayPred()
- segmentImages()
- onlineSegmentation()
'''

from numpy import expand_dims, uint8, ndarray, zeros_like
from generalUtensils import getTimeStamp
import matplotlib.pyplot as plt
from random import randint
from dataPreparation import resizeSingleFrame
from postProcessing import measurementVB, writeCSV, plotWearCurve, outlierDetection, plotWearCurveLOWESS
from sklearn.preprocessing import normalize
from cv2 import resize, addWeighted, cvtColor, COLOR_BGR2GRAY, imwrite, imread, IMREAD_GRAYSCALE
from os import makedirs, getcwd, listdir
from os.path import join

def displayPred(testImg: ndarray, groundTruth: ndarray, pred) -> None:
    plt.figure(figsize=(16,8))
    plt.subplot(131) # plot original image
    plt.title('Testing Image')
    plt.imshow(testImg[0], cmap='gray')
    plt.subplot(132) # plot true mask of image
    plt.title('Predicted Mask')
    plt.imshow(pred, cmap='gray')
    plt.show()

def segmentImages(model, testData: dict, randomize: bool = True) -> None:
    ''' Testing model on random images. Takes in test data and randomization command.
    Prints out images (pyplot).
    If "randomize" = True: pick random image from stack.
    '''
    if randomize: # maybe change to: pick random or take in single
        idx = randint(0, len(testData['xTest'])) # create random index
        targetImg = testData['xTest'][idx] # select random image
        groundTruth = testData['yTest'][idx]
    else: idx = 0
    targetImgNorm = targetImg[:,:,0][:,:,None] 
    targetImgInput = expand_dims(targetImgNorm, 0) # expand dimension
    targetImgResize = resizeSingleFrame(frame=targetImgInput, aspectRatio=(512,512))
    pred = (model.predict(targetImgResize)[0,:,:,0] > 0.2).astype(uint8)
    displayPred(testImg=targetImg, groundTruth=groundTruth, pred=pred) # displays results
    maxVB = measurementVB(frame=pred)
    print('Maximum VB: ', maxVB)

def segmentDataStack(dataPath, model, nrEdges, savePath):
    imgName = []
    # print('Image Stack Length: ', len(imageStack))
    resultFolder = join(savePath, 'results', str(getTimeStamp())) # Define the path to save the CSV file
    makedirs(resultFolder)
    # predFolder = join(resultFolder,'pred')
    # makedirs(predFolder)
    # for i in range(len(imageStack)): 
    #     pred = predictSingleFrame(imageStack[i], model)
    #     maskStack.append(pred)
    #     imwrite(join(predFolder,'pred_'+str(i)+'.tiff'), pred)
    # print('Length Result Array: ',len(maskStack))
    resultsVBMax, resultFolder, resultFile = wearDetectionStack(dataPath=dataPath, model=model, nrEdges=nrEdges, resultFolder=resultFolder)
    wearCurve = plotWearCurve(filePath=resultFile, resultFolder=resultFolder)
    wearCurve = outlierDetection(filePath=resultFile,resultFolder=resultFolder)
    wearCurve = plotWearCurveLOWESS(filePath=resultFile, resultFolder=resultFolder)
    return wearCurve

def wearDetectionStack(dataPath, model, nrEdges, resultFolder):
    ''' Takes in array of masks. '''
    resultsVBMax = []
    imgName = []
    fitFolder = join(resultFolder, 'fit')
    makedirs(fitFolder)
    predFolder = join(resultFolder,'pred')
    makedirs(predFolder)
    print('Estimate maximum VB')
    for filename in listdir(dataPath):
        imgName.append(filename)
        img = imread(join(dataPath, filename), IMREAD_GRAYSCALE)
        pred = predictSingleFrame(img, model)
        imwrite(join(predFolder,'pred_'+filename+'.tiff'), pred)
        sampleVBMax = measurementVB(frame=pred, saveFolder=resultFolder, filename=filename)
        resultsVBMax.append(sampleVBMax)
        print('\nFilename: ', filename, '\nVB max: ',sampleVBMax)
    resultFolder, resultFile = writeCSV(resultsVBMax=resultsVBMax, resultFolder=resultFolder)
    return resultsVBMax, resultFolder, resultFile

def predictSingleFrame(frame: ndarray, model):
    if len(frame.shape) > 2: frame = resize(frame, (frame.shape[0], frame.shape[1]))
    frameNorm = expand_dims(normalize(frame), 2) # expected shape: (512, 512, 1)
    frameNormExpand = expand_dims(frameNorm, 0) # expected shape: (1, 512, 512, 1): format for model prediction
    predMask = (model.predict(frameNormExpand)[0,:,:,0]>0.2).astype(uint8) # expected shape: (512, 512)
    return predMask*255

def singleImageSegmentation(image: ndarray, model) -> ndarray:
    ''' Online Segmentation of images. Takes in single image and model to segment with.
    Returns predicted mask of the input image.
    
    Image needs preprocessing! '''
    print('Segmenting Image. This might take a while.') # implement tqdm or something else to track segmenting process
    imageResized = resizeSingleFrame(frame=image, aspectRatio=(512,512))
    mask = predictSingleFrame(frame=imageResized, model=model)
    maxVB = measurementVB(frame=mask, saveFolder=join(getcwd(), 'results'))
    print('Maximum VB: ', maxVB)
    return mask, maxVB

def videoSegmentation(frame, model):
    ''' Takes in single frame of video stream, model for prediction and desired aspect ratio (latter optional)
    Returns single frame overlayed with image prediction. '''
    print('Video Segmentation Init Frame: ', frame.shape)
    frameGrayScaled = cvtColor(frame, COLOR_BGR2GRAY)
    frameDownsized = resize(frameGrayScaled, (512, 512)) # to fit model prediction
    frameNorm = expand_dims(normalize(frameDownsized), 2) # expected shape: (512, 512, 1)
    frameNormExpand = expand_dims(frameNorm, 0) # expected shape: (1, 512, 512, 1): format for model prediction
    predMask = (model.predict(frameNormExpand)[0,:,:,0]>0.2).astype(uint8) # expected shape: (512, 512)
    predMaskUpsize = resize(predMask, (frame.shape[1], frame.shape[0])) # expected shape: (frameHeight, frameWidth)
    redMask = zeros_like(frame) # expected shape: frame.shape
    redMask[predMaskUpsize==1]=[0,0,255]
    overlayFrame = addWeighted(frame, 1.0, redMask, 0.5, 0)
    print('Overlay Frame: ',overlayFrame.shape)
    return overlayFrame

