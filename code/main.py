import os, logging, sys, base64, eel
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

from generalUtensils import setupData, getTestData, loadCurModel, imageReader, saveImage
from evaluateTraining import evalTraining
from onlinePreprocessing import preProcImage
from offlinePreprocessing import offlinePreProc, resizeImage
from segmentation import segmentImages, onlineSegmentation, displayPred
from camera_system import cameraSetup, videoStream, reformatImage
from modelTraining import saveCurModel, trainCurModel, createModel
from header import model_save_path, data_path, currentModel#, currentImage # NEED TO BE ALTERED

# --------------- VIDEO STREAM ------------------------
@eel.expose()
def videoFeed():
    ''' Set up video stream. '''
    global cam
    cam = cameraSetup() if cameraSetup() else False
    if cam == False: return # shut down stream if camera is not setup
    y = videoStream(camera=cam)
    for each in y:
        blob = base64.b64encode(each)
        blob = blob.decode('utf-8')
        eel.updateImageSrc(blob)() # expose video stream to HTML interface

@eel.expose()
def takePicture():
    ''' Take picture with the camera. '''
    cameraConnection = False # shut down camera.
    _, image = cam.save_frame() # returns filename, image
    blob = reformatImage(image=image)
    eel.updatePicture(blob)() # expose picture to HTML interface

# ------------ LOADING & SAVING FUNCTIONS ----------------

@eel.expose()
def loadImage(path: str):
    ''' Load image from input path. Path passed from HMTL surface.'''
    global currentImage # current... = temporarily stored
    currentImage = imageReader(targetPath=path, segment=True) # single file
    print('LOADED IMAGE: shape: ', currentImage.shape)
    transferImage = currentImage.copy()
    # imshow("Current Image", currentImage)
    plt.imshow(transferImage[0])
    plt.show()
    print(currentImage)
    blob = reformatImage(transferImage[0])
    eel.updatePicture(blob)() # expose picture to HTML interface

@eel.expose()
def saveImages():
    ''' Save current image. EMPTY AT THE MOMENT.'''
    return

@eel.expose()
def loadModel(path: str):
    ''' Load Model from path to saved model file. '''
    global currentModel
    currentModel = loadCurModel(path=path) # expose current model to other functions

@eel.expose()
def saveModel():
    ''' Save current model to selected directory. DIRECTORY SELECTION MISSING. '''
    saveCurModel(model=currentModel, model_path=model_save_path) # PATH NEEDS TO BE ALTERED


@eel.expose()
def getFile(elementID):
    ''' Select single file over Tkinter interface and file explorer. File is assigned an "elementID" in HTML interface. '''
    root = Tk()
    filename = filedialog.askopenfilename(initialdir=os.getcwd())
    root.destroy()
    eel.updateDirectoryName(filename,elementID)()

@eel.expose()
def getDirectory(elementID):
    ''' Select directory over Tkinter interface and file explorer. Directory is assigned an "elementID" in HTML interface. '''
    # opening tkinter interface to file explorer; function to select directory
    root = Tk()
    directory = filedialog.askdirectory(initialdir=os.getcwd())
    root.destroy()
    eel.updateDirectoryName(directory,elementID)()

# -------------- PREPROCESSING -----------------------

# Empty Functions: (previously used, for now discarded)
@eel.expose()
def init_reshape(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, x4: int,y4: int): return
def reshape_image(): return
def auto_crop(): return
def change_mask_extension(parameters): return

@eel.expose()
def changeImageShape(parameters):
    ''' Reshaping images. In HTML interface an "ImageDirectoryGoal_dir" is selected.
    This is the path to the project file!'''
    parDict = {entry[0]: entry[1] for entry in parameters}
    rawData = setupData(projectPath=parDict['ImageDirectoryGoal_dir'], split=False)
    images = resizeImage(parameters=parDict, data=rawData[0])
    masks = resizeImage(parameters=parDict, data=rawData[1])
    print(parDict)
    saveImage(pathTarget=parDict['ImageDirectoryGoal_dir'],image=images, token='resized')
    saveImage(pathTarget=parDict['ImageDirectoryGoal_dir'],image=masks, token='resized', imgData=False)


# --------------- MODEL TRAINING ----------------------

# Empty Functions: (previously used, for now discarded)
def trainTestSplit(parameters): return 

@eel.expose()
def trainModel(parameters): # CHECK IF PARAMETERS HAVE THE RIGHT FORM
    '''Execute model training. Parameters are passed from the HTML interface. '''
    global history, trainData, currentModel # expose training parameters to other functions
    parDict = {entry[0]: entry[1] for entry in parameters} # convert to dictionary
    trainData = setupData(parameters[2][1], split=True) # gather data from project name 
    currentModel, history = trainCurModel(model=createModel(), trainData=trainData, parDict=parDict)
    return currentModel, history

@eel.expose()
def evaluateModel(): # MISSING EEL EXTENSION
    ''' Evaluate model training. HAS NO EEL / HTML EXTENSION YET.
    Not sure, if extension is needed, or if the results can be displayed in a different way. '''
    evalTraining(modelTrained=currentModel, history=history, testData=getTestData(trainData)) # evaluate training
    # eel.outputMetric(metricName, metrics[metricName]) # TO BE IMPLEMENTED


# --------------- IMAGE SEGMENTATION -----------------

@eel.expose()
def segmentImage():
    ''' Segment image with online segmentation'''
    global currentImage, currentModel
    pred = onlineSegmentation(image=currentImage, model=currentModel)
    displayPred(testImg=currentImage, groundTruth=None, pred=pred)

# --------------- APP EXECUTION ----------------------

def startApp():
    ''' Start HTML application. '''
    try:
        BASEDIR = os.path.dirname(os.path.abspath(__file__))    # static
        print('\n---- STARTING HTML APPLICATION ----')
        print('Base Directory:\t\t', BASEDIR)
        WEBDIR = os.path.join(BASEDIR, 'frontend')
        print('Web Dir:\t\t',WEBDIR,'\n') # initialize HTML interface in the 'frontend' folder
        eel.init(WEBDIR) 
        logging.info("App Started") # starting the HTML interface over 'index.html' file; 
        eel.start("index.html", mode='Chrome')# change "mode" depending on browser to use application in
    except Exception as e: # throw error if HTML setup is faulty
        err_msg = 'Could not launch a local server'
        logging.error('{}\n{}'.format(err_msg, e.args))
        logging.info('Closing App')
        sys.exit()

# -------------- EMPTY FUNCTIONS ---------------------

def dataPreparation():
    '''Offline Data Preprocessing'''
    global projectPath
    offlinePreProc(projectPath=projectPath)


def segment():
    ''' Segment image with offline segmentation. '''
    trainData = getTestData(setupData(data_path, split=True))
    segmentImages(testData=trainData)


if __name__ == '__main__':
    ''' Start script '''
    startApp()
