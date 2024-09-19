''' ## Funtions for Training of New Model

Accessible Functions:
- createModel()
- trainModel()
- saveModel()
'''

import evaluateTraining
from header import IMG_SHAPE,MODEL_FORMAT
from header import currentModel # MAYBE NEED TO BE ADJUSTED (no overwriting possible)
from modelArchitecture import UNet
from os.path import join
from generalUtensils import getTimeStamp
from tensorflow import keras
from keras.models import Model


def createModel() -> Model:
    ''' Create empty model. MODEL ARCHITECTURE IS FIXED (maybe function to select between different architectures). '''
    model = UNet.unet_ehsan(img_shape=IMG_SHAPE)
    return model

def trainCurModel(model: Model, trainData: dict,  parDict: dict) -> Model:
    ''' Train model on training data. Takes in model, training data, and training parameters.
    Return trained model and training history'''
    history = model.fit(x=trainData['xTrain'], 
                        y=trainData['yTrain'], 
                        batch_size=int(parDict['BatchSize_int']),
                        verbose = True, # by default True; PARAMETER CAN BE ADDED
                        epochs=int(parDict['Epochs_int']), 
                        validation_data=(trainData['xTest'], trainData['yTest']), 
                        shuffle=bool(parDict['ShuffleTraining_bool']))
    return model, history

def saveCurModel(model: Model, modelPath: str) -> bool:
    ''' Save model to model directory with current timestamp.'''
    timeStamp = getTimeStamp() # name model according to time stamp
    dirPath = join(modelPath,'Model_Training_'+timeStamp+MODEL_FORMAT)
    print('\nSaving Model to:\n'+dirPath+'\n')
    model.save(dirPath)
    return True
