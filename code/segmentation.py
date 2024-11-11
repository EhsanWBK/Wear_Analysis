from numpy import expand_dims, uint8, ndarray, zeros_like
from generalUtensils import getTimeStamp
import matplotlib.pyplot as plt
from random import randint
from dataPreparation import resizeSingleFrame
from postProcessing import measurementVB, writeCSV, plotWearCurve, outlierDetection
from sklearn.preprocessing import normalize
from cv2 import resize, addWeighted, cvtColor, COLOR_BGR2GRAY, imwrite, imread, IMREAD_GRAYSCALE
from os import makedirs, getcwd, listdir
from os.path import join, exists

def segmentDataStack(dataPath, model, nrEdges, savePath):
    resultFolder = join(savePath, 'results', str(getTimeStamp())) # Define the path to save the CSV file
    if not exists(resultFolder): makedirs(resultFolder)
    _, resultFolder, resultFile = wearDetectionStack(dataPath=dataPath, model=model, nrEdges=nrEdges, resultFolder=resultFolder)
    wearCurve = plotWearCurve(filePath=resultFile, resultFolder=resultFolder)
    wearCurve = outlierDetection(filePath=resultFile,resultFolder=resultFolder)
    return wearCurve

def wearDetectionStack(dataPath, model, nrEdges, resultFolder):
    resultsVBMax = []; imgName = []
    fitFolder = join(resultFolder, 'fit')
    if not exists(fitFolder): makedirs(fitFolder)
    predFolder = join(resultFolder,'pred')
    if not exists(predFolder): makedirs(predFolder)
    for filename in listdir(dataPath):
        imgName.append(filename)
        img = imread(join(dataPath, filename), IMREAD_GRAYSCALE)
        pred = predictSingleFrame(img, model)
        imwrite(join(predFolder,'pred_'+filename+'.tiff'), pred)
        sampleVBMax = measurementVB(frame=pred, saveFolder=resultFolder, filename=filename)
        resultsVBMax.append(sampleVBMax); print('Filename: ', filename, '\nVB max: ',sampleVBMax,'\n')
    resultFolder, resultFile = writeCSV(resultsVBMax=resultsVBMax, resultFolder=resultFolder)
    return resultsVBMax, resultFolder, resultFile

def predictSingleFrame(frame: ndarray, model):
    if len(frame.shape) > 2: frame = resize(frame, (frame.shape[0], frame.shape[1]))
    frameNorm = expand_dims(normalize(frame), 2) # expected shape: (512, 512, 1)
    frameNormExpand = expand_dims(frameNorm, 0) # expected shape: (1, 512, 512, 1): format for model prediction
    predMask = (model.predict(frameNormExpand)[0,:,:,0]>0.2).astype(uint8) # expected shape: (512, 512)
    return predMask*255

def singleImageSegmentation(image: ndarray, model) -> ndarray:
    ''' Online Segmentation of images. Takes in single image and model to segment with. Returns predicted mask of the input image.'''
    imageResized = resizeSingleFrame(frame=image, aspectRatio=(512,512))
    mask = predictSingleFrame(frame=imageResized, model=model)
    maxVB = measurementVB(frame=mask, saveFolder=join(getcwd(), 'results')); print('Maximum VB: ', maxVB)
    return mask, maxVB

def videoSegmentation(frame, model):
    ''' Takes in single frame of video stream, model for prediction and desired aspect ratio (latter optional)
    Returns single frame overlayed with image prediction. '''
    frameGrayScaled = cvtColor(frame, COLOR_BGR2GRAY)
    frameDownsized = resize(frameGrayScaled, (512, 512)) # to fit model prediction
    frameNorm = expand_dims(normalize(frameDownsized), 2) # expected shape: (512, 512, 1)
    frameNormExpand = expand_dims(frameNorm, 0) # expected shape: (1, 512, 512, 1): format for model prediction
    predMask = (model.predict(frameNormExpand)[0,:,:,0]>0.2).astype(uint8) # expected shape: (512, 512)
    predMaskUpsize = resize(predMask, (frame.shape[1], frame.shape[0])) # expected shape: (frameHeight, frameWidth)
    redMask = zeros_like(frame) # expected shape: frame.shape
    redMask[predMaskUpsize==1]=[0,0,255]
    overlayFrame = addWeighted(frame, 1.0, redMask, 0.5, 0)
    return overlayFrame
