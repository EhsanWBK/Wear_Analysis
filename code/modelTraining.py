''' ## Funtions for Training of New Model

Accessible Functions:
- createModel()
- trainModel()
- saveModel()
'''

from modelArchitecture import UNet
from dataPreparation import resizeAll
from generalUtensils import setupData, getTimeStamp
from header import CWD

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from os.path import join
from tensorboard import program
from matplotlib import pyplot as plt
from numpy import expand_dims
from pandas import DataFrame

import webbrowser

# ======== Model Initialization ========

def createModel(inputShape, par) -> Model:
    model = UNet.unet_ehsan(inputShape=inputShape)
    # model = UNet.unet_first_iteration(inputShape=inputShape,n_classes=int(par['nrChannels']))
    return model

# ======== Model Training ========

def trainCurModel(par: dict) -> Model:
    ''' Train model on training data. Takes in model, training data, and training parameters.
    Return trained model and training history'''
    global epochs, batch_size, aspectRatio, channel, log_dir

    inputShape = (int(par['imageHeight']), int(par['imageWidth']), int(par['nrChannels']))
    aspectRatio = inputShape[:2]
    channel = inputShape[2]
    print('Aspect Ration: ',aspectRatio,' and Number of Channels: ', channel)

    modelName = str(par['modelName'])
    projectPath = str(par['trainingImgDir'])
    modelSavePath = str(par['modelSavingDir'])

    batch_size=int(par['batchSize'])
    epochs=int(par['nrEpochs'])
    shuffle=bool(par['shuffleTrain'])
    augmentation = bool(par['selectAug'])

    trainData = setupData(projectPath=projectPath, par=par, split=True, token='final') 
    model = createModel(inputShape=inputShape, par=par)
    
    # transfer learning

    # Callback Parameter
    monitor = 'val_loss'
    earlyStopPatience = int(par['earlyStopping'])
    checkPointPath = join(modelSavePath,modelName +'_best')
    log_dir = join(CWD,'logs','fit',str(getTimeStamp()))

    # Callback Setup
    earlyStops = EarlyStopping(monitor=monitor, patience=earlyStopPatience, restore_best_weights=True)
    checkpoints = ModelCheckpoint(filepath=checkPointPath, monitor=monitor, verbose=True, save_best_only=True, mode='auto', save_freq='epoch')
    tensorboardCallback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [earlyStops, checkpoints, tensorboardCallback]

    x ,y, _ = resizeAll(img=trainData['xTrain'], aspectRatio=aspectRatio, mask=trainData['yTrain']/255.0, saveProgress=False)
    print('Input Shape: ', x.shape)
    if augmentation: model, history = augementImage(model=model, trainData=trainData, par=par, callbacks=callbacks)
    else: history = model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, validation_data=(trainData['xTest'], trainData['yTest']), 
                            shuffle=shuffle, callbacks=callbacks, verbose = True)

    print('Model Training Finished successfully.')
    return model, history

def augementImage(model: Model, trainData: dict, par: dict, callbacks: list):
    ''' Takes in Training Data Set and returns Image and Mask Data Generator. '''
    shuffle = bool(par['randomSelection'])
    seed = int(par['randomState'])

    arguments = {
        'rotation_range':       int(par['rotationRange']),
        'width_shift_range':    float(par['widthShiftRange']),
        'height_shift_range':   float(par['heigthShiftRange']),
        'zoom_range':           float(par['zoomRange']),
        'horizontal_flip':      bool(par['horizontalFlip']),
        'vertical_flip':        bool(par['verticalFlip']),
        'validation_split':     float(par['validationSize'])
    }

    datagenImage = ImageDataGenerator(**arguments)
    datagenMask = ImageDataGenerator(**arguments)

    xTrain ,yTrain, _ = resizeAll(img=trainData['xTrain'], aspectRatio=aspectRatio, mask=trainData['yTrain'], saveProgress=False)
    xTest ,yTest, _ = resizeAll(img=trainData['xTest'], aspectRatio=aspectRatio, mask=trainData['yTest'], saveProgress=False)

    xTrain = expand_dims(xTrain, axis=-1)
    yTrain = expand_dims(yTrain, axis=-1)
    xTest = expand_dims(xTest, axis=-1)
    yTest = expand_dims(yTest, axis=-1)
    
    imageGenTrain = datagenImage.flow(x=xTrain, batch_size=batch_size, shuffle=shuffle, seed=seed,
        subset='training'
    )
    maskGenTrain = datagenMask.flow(x=yTrain, batch_size=batch_size, shuffle=shuffle, seed=seed,
        subset='training'
    )
    imageGenVal = datagenImage.flow(x=xTest, batch_size=batch_size, shuffle=shuffle, seed=seed,
        subset='validation'
    )
    maskGenVal = datagenMask.flow(x=yTest, batch_size=batch_size, shuffle=shuffle, seed=seed,
        subset='validation'
    )

    history = model.fit(zip(imageGenTrain, maskGenTrain), validation_data=zip(imageGenVal, maskGenVal), epochs = epochs, callbacks = callbacks, verbose=True )
    return model, history

# ======== Model Training Evaluation ========

def evalModelTraining(history):
    fig ,ax = plt.subplots(3, 1, figsize=(8,12))
    epochs = range(1, len(loss)+1)

    # Training Loss and Validation Loss
    loss = history.history['loss']
    val_loss = history.history['val_los']
    ax[0].plot(epochs, loss, 'y', label='Training Loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation Loss')  
    ax[0].set_title('Training adn Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')

    # Training Accuracy and Validation Accuracy
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    ax[1].plot(epochs, acc, 'y', label='Training Accuracy')
    ax[1].plot(epochs, val_acc, 'r', label='Validation Accuracy')  
    ax[1].set_title('Training Accuracy and Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')

    # Training IoU and Validation IoU
    iou = history.history['one_hot_io_u']
    val_iou = history.history['val_one_hot_io_u']
    ax[2].plot(epochs, iou, 'y', label='Training IoU')
    ax[2].plot(epochs, val_iou, 'r', label='Validation IoU')  
    ax[2].set_title('IoU (Intersetion over Union)')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()

def displayTensorboard(url):
    try:
        tb = program.Tensorboard()
        tb.configure(argv=[None, '--logdir', log_dir])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")
        webbrowser.open(url)
    except Exception as e:
        print('Cannot start tensorboard: ',e)

def saveHistory(path, history):
    df_history = DataFrame(history.history)
    hist_csv_file = join(path, 'hist_'+getTimeStamp()+'.csv')
    with open(hist_csv_file, mode='w') as f:
        df_history.to_csv(f)