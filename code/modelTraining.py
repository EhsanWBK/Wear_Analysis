from modelArchitecture import UNet
from dataPreparation import resizeAll
from generalUtensils import setupData, getTimeStamp, setupAugemented
from header import CWD

from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from os.path import join
from matplotlib import pyplot as plt
from pandas import DataFrame


# ======== Model Initialization ========

def createModel(inputShape, par) -> Model:
    return UNet.unet_ehsan(inputShape=inputShape)

# ======== Model Training ========

def trainCurModel(par: dict) -> Model:
    ''' Train model on training data. Takes in model, training data, and training parameters.
    Return trained model and training history'''
    global epochs, batch_size, aspectRatio, channel, log_dir
    inputShape = (int(par['imageHeight']), int(par['imageWidth']), int(par['nrChannels']))
    aspectRatio = inputShape[:2]; channel = inputShape[2]
    modelName = str(par['modelName']); projectPath = str(par['trainingImgDir']); modelSavePath = str(par['modelSavingDir'])
    batch_size=int(par['batchSize']); epochs=int(par['nrEpochs']); shuffle=bool(par['shuffleTrain']); augmentation = bool(par['selectAug'])
    if augmentation: trainData = setupAugemented(projectPath=projectPath, parDic=par, split=True)
    else: trainData = setupData(projectPath=projectPath, par=par, split=True, token='final') 
    model = createModel(inputShape=inputShape, par=par)

    monitor = 'val_loss'; earlyStopPatience = int(par['earlyStopping']) # Callback Parameter
    checkPointPath = join(modelSavePath,modelName +'_best')
    log_dir = join(CWD,'logs','fit',str(getTimeStamp()))
    earlyStops = EarlyStopping(monitor=monitor, patience=earlyStopPatience, restore_best_weights=True)
    checkpoints = ModelCheckpoint(filepath=checkPointPath, monitor=monitor, verbose=True, save_best_only=True, mode='auto', save_freq='epoch')
    callbacks = [earlyStops, checkpoints]

    x ,y, _ = resizeAll(img=trainData['xTrain'], aspectRatio=aspectRatio, mask=trainData['yTrain']/255.0, saveProgress=False)
    history = model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, validation_data=(trainData['xTest'], trainData['yTest']), shuffle=shuffle, callbacks=callbacks, verbose = True)
    print('Model Training Finished successfully.')
    return model, history


# ======== Model Training Evaluation ========

def evalModelTraining(history):
    _ ,ax = plt.subplots(3, 1, figsize=(8,12))
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
    plt.tight_layout(); plt.show()

def saveHistory(path, history):
    df_history = DataFrame(history.history)
    hist_csv_file = join(path, 'hist_'+getTimeStamp()+'.csv')
    with open(hist_csv_file, mode='w') as f: df_history.to_csv(f)