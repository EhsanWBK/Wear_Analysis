import eel
from tkinter import Tk, filedialog
from time import sleep
from datetime import datetime
from os import makedirs
from os.path import join, exists
from sys import exit
from threading import Event, Thread
from cv2 import imwrite
from copy import deepcopy

from generalUtensils import loadCurModel, imageReader, reformatFrame, saveCurModel, pathCreator, getTimeStamp, saveTrigger
from dataPreparation import preProcStart, preProcFromCamera, preProcForSegment
from segmentation import singleImageSegmentation, videoSegmentation, segmentDataStack
from cameraSystem import Baumer
from modelTraining import trainCurModel, saveHistory
from header import *

#  =========================================
#  	         Multithreading	Setup
#  =========================================

htmlClosed = Event(); pictureEvent = Event(); videoEvent = Event(); stopEvent = Event(); streamSegEvent = Event(); triggerEvent = Event()
videoCam = Baumer()

# Thread 1:
def startCamera(sharedArray, stopEvent):
    ''' Starting the OPC UA Client for the Camera.'''
    global streamFrame
    print('\n----------------------- STARTING CAMERA -----------------------')
    try:
        while not stopEvent.is_set(): 
            camBuffer = videoCam.getFrame()
            try:
                sharedArray[:] = camBuffer; streamFrame = sharedArray
            except: pass
            # if videoCam.triggerStatus():  pass # image is in trigger mode
    finally: videoCam.stopCam()

# Thread 2:
def streamVid(event, stopEvent):
    print('\t- Video Thread Set Up.')
    while not stopEvent.is_set():
        event.wait()
        if stopEvent.is_set(): return
        print('Start Streaming Data')
        while event.is_set() and not stopEvent.is_set(): # to stop stream: call videoEvent.clear() outside of this function
            blob = reformatFrame(frame=streamFrame)
            if event.is_set(): eel.updateCanvas1(blob)() # implement timeout function OR delete cache in eel, when html is closed.
        print('Stopped Streaming Data.')
        event.clear()

# Thread 3:
def sendPicture(sharedArray, event, stopEvent):
    global freezeFrame
    print('\t- Picture Thread Set Up.')
    while not stopEvent.is_set(): 
        event.wait()
        if stopEvent.is_set(): return  
        event.clear()    
        freezeFrame = preProcFromCamera(streamFrame)
        blob = reformatFrame(frame=freezeFrame)
        eel.updateCanvas1(blob)()

# Thread 4:
def streamSeg(event, stopEvent):
    print('\t- Video Segmentation Thread Set Up.')
    while not stopEvent.is_set():
        event.wait()
        if stopEvent.is_set(): return
        while event.is_set() and not stopEvent.is_set(): # to stop stream: call videoEvent.clear() outside of this function
            overlayFrame = videoSegmentation(frame=streamFrame, model=currentModel)
            blob = reformatFrame(frame=overlayFrame)
            if event.is_set(): eel.updateCanvas2(blob)() # implement timeout function OR delete cache in eel, when html is closed.
        event.clear()

# Thread 5:
def observeTrigger(event, stopEvent):
    print('\t- Trigger Observation Thread Set Up.')
    triggerTemp = []
    while not stopEvent.is_set():
        event.wait()
        if stopEvent.is_set():return
        print('Start Checking Trigger')
        if event.is_set() and not stopEvent.is_set(): videoCam.triggerModeOn()
        while event.is_set() and not stopEvent.is_set():
            if videoCam.triggerStatus(): 
                print('Trigger detected.')
                frame = videoCam.getFrame(); triggerTemp.append(deepcopy(frame))
                blob = reformatFrame(frame=frame)
                eel.updateCanvas2(blob)()
            if not videoCam.triggerStatus() and triggerTemp != []:
                saveFolder = saveTrigger(triggerTemp); triggerTemp = []
                try: segmentDataStack(dataPath=saveFolder, model=currentModel, savePath=saveFolder)
                except: print('To Segment Images load model. ')
        videoCam.triggerModeOff(); event.clear()
    print('Trigger Observation to be terminated')

#  =========================================
#  	       HTML Interface Functions		
#  =========================================

# ------------------
#    HTML SETUP
# ------------------

def startHTML():
    ''' Start HTML application. '''
    try:
        BASEDIR = CWD
        print('\n----------------------- STARTING HTML APPLICATION -----------------------')
        print('Base Directory:\t\t', BASEDIR)
        WEBDIR = join(BASEDIR, 'code', 'frontend'); print('Web Dir:\t\t',WEBDIR,'\n') 
        eel.init(WEBDIR); sleep(1); eel.start("index.html", mode='Chrome')
    except Exception as e:exit()

@eel.expose()
def windowClosed():
    print('\n----------------------- CLOSING HTML WINDOW -----------------------')
    print('\t- Clearing Video Stream Event.'); videoEvent.clear(); print('\t- Video Stream Event is cleared.')
    sleep(2); streamSegEvent.clear(); print('\t- Video Segmentation Event is cleared.')
    sleep(2); triggerEvent.clear(); print('\t- Trigger Event is cleared.')
    sleep(2); print('\t- Set HMTL Close Event'); htmlClosed.set(); print('\t- HTML can be closed now.')

# ----------------------------------
#        Interface Functions
# ----------------------------------

@eel.expose()
def getFile(elementID):
    path = CWD
    if elementID == 'imageInput': path=SINGLE_DATA_PATH
    if elementID == 'modelDirPath': path=SAVE_MODEL_PATH
    root = Tk(); root.attributes("-topmost", True)
    filename = filedialog.askopenfilename(initialdir=path) # CHANGE TO DATA FILE
    root.destroy()
    eel.updateDirectoryName(filename,elementID)()
    if elementID == 'imageInput': loadImage(filename)
    elif elementID == 'modelDirPath': loadModel(filename)
    else: print('Error Loading File.')

@eel.expose()
def getDirectory(elementID):
    path = CWD
    if elementID == 'imageInput' or elementID == 'dataDirectory' or elementID == 'trainingImgDir': path = TRAIN_DATA_PATH
    if elementID =='SaveResultDir': path = SAVE_RES_PATH
    if elementID == 'modelSavingDir': path = SAVE_MODEL_PATH
    root = Tk(); root.attributes("-topmost", True)
    directory = filedialog.askdirectory(initialdir=path)
    root.destroy()
    eel.updateDirectoryName(directory,elementID)()

# ======== Data Preparation ========

@eel.expose()
def preProcSteps(argument, parameters):
    projectPath = parameters[0] if len(parameters) < 3 else str(parameters[0])
    aspectRatio = None if len(parameters) < 3 else (int(parameters[1]), int(parameters[2]))
    print('\nStarting Preprocessing for ', argument[0]); print('Project Path: ', projectPath)
    preProcStart(argument=argument, projectPath=projectPath, aspectRatio=aspectRatio)

# ======== Model Training ========

@eel.expose()
def trainModel(par):
    global history, trainedModel
    print('Extracted Parameter Dictionary: ', par)
    trainedModel, history = trainCurModel(par=par)
    modelSavePath = join(str(par['modelSavingDir']), str(par['modelName']))
    saveModel(modelSavePath); saveHistory(modelSavePath, history)
    eel.modelTrained()
    return trainedModel, history

@eel.expose()
def saveModel(path):
    success = False
    if trainedModel is not None: success = saveCurModel(model=trainedModel, modelPath=path)
    if success: print('Successfully Saved the model.')

# ======== Image Segmentation ========

@eel.expose()
def videoFeed():
    if videoEvent.set():videoEvent.clear()
    else: videoEvent.set()

@eel.expose()
def stopVideo():
    if videoEvent.is_set(): videoEvent.clear()
    if streamSegEvent.is_set(): streamSegEvent.clear()
    else: print('No Video Stream to be stopped.')

@eel.expose()
def setTrigger():
    if triggerEvent.set(): 
        print('Stopping Trigger');# videoCam.stopTrigger(); 
        triggerEvent.clear()
    else: print('Start Trigger'); triggerEvent.set()

@eel.expose()
def takePicture():
    videoEvent.clear(); print('Cleared Video Event.')
    pictureEvent.set()

@eel.expose()
def loadImage(path: str):
    ''' Load image from input path. Path passed from HMTL surface.'''
    global freezeFrame # current... = temporarily stored
    freezeFrame, _ = imageReader(targetPath=path, segment=True) # single file
    print('Loading image from directory (with shape): ', freezeFrame.shape)
    transferImage = freezeFrame.copy()
    blob = reformatFrame(transferImage[0])
    eel.updateCanvas1(blob)() # expose picture to HTML interface

@eel.expose()
def saveSegResult(path: str):
    curTimeStr = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    filename = join(path, curTimeStr) + '.jpg'
    imwrite(filename, currentResultImg)

@eel.expose()
def loadModel(path: str):
    global currentModel
    currentModel = loadCurModel(path=path) # expose current model to other functions

@eel.expose()
def segmentImage():
    global currentResultImg
    currentResultImg, _ = singleImageSegmentation(image=freezeFrame, model=currentModel)
    blob = reformatFrame(currentResultImg)
    eel.updateCanvas2(blob)()

@eel.expose()
def segmentStack(pathProj, nrEdges):
    print('Segmenting Data Stack.')
    pathSeg, _ = pathCreator(pathProj, grabData=True, token='seg')
    if not exists(path=pathSeg):
        print('Preprocessing Images for Segmentation') 
        makedirs(pathSeg)
        pathRaw, _ = pathCreator(pathProj, grabData=True)
        rawImg, rawFileName = imageReader(pathRaw)
        preProcForSegment(imgArray=rawImg, projectPath=pathProj, fileNames=[rawFileName])
    wearCurve = segmentDataStack(dataPath=pathSeg, model=currentModel, nrEdges=int(nrEdges), savePath=pathProj)
    blob = reformatFrame(wearCurve)
    eel.updateCanvas2(blob)()
    
@eel.expose()
def segmentVideo():
    if not streamSegEvent.is_set(): print('\nStarting Video Segmentation.'); streamSegEvent.set()  
    else: print('\nStopping Video Segmentation.'); streamSegEvent.clear()

#  =========================================
#  	   General Setup and Initialization		
#  =========================================

def setup():
    print('\n----------------------- STARTING THREADS -----------------------')
    cameraThread.start(); sleep(5)
    pictureThread.start(); sleep(1)
    videoThread.start(); sleep(1)
    onlineSegThread.start(); sleep(1)
    triggerThread.start(); sleep(1)
    htmlThread.start()

def shutdown():
    print('\n----------------------- SHUTTING DOWN PROGRAM -----------------------')
    stopEvent.set()
    videoEvent.set(); streamSegEvent.set(); triggerEvent.set(); print('\t- Released Video Event.')
    pictureEvent.set(); cameraThread.join(timeout=1); print('\t- Released Picture Event.')
    pictureThread.join(timeout=1); print('\t- Stopped Picture Thread.')
    videoThread.join(timeout=1); print('\t- Stopped Video Thread.')
    onlineSegThread.join(timeout=1); print('\t- Stopped Online Segmentation Thread.')
    triggerThread.join(timeout=-1); print('\t- Stopped Trigger Observation Thread.')
    print('\t- Stopped all Threads.')

if __name__ == '__main__':
    cameraThread = Thread(target=startCamera, args=(IMG_ARRAY, stopEvent)) # Thread 1 
    pictureThread = Thread(target=sendPicture, args=(IMG_ARRAY, pictureEvent, stopEvent)) # Thread 2
    videoThread = Thread(target=streamVid, args=(videoEvent, stopEvent)) # Thread 3
    onlineSegThread = Thread(target=streamSeg, args=(streamSegEvent, stopEvent)) # Thread 4
    htmlThread = Thread(target=startHTML) # Thread 5
    triggerThread = Thread(target=observeTrigger, args=(triggerEvent, stopEvent)) # Thread 6
    setup()
    htmlClosed.wait(); htmlThread.join(); shutdown() # shutting down system
