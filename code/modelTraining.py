from modelArchitecture import UNet
from dataPreparation import resizeAll
from generalUtensils import setupData, getTimeStamp, setupAugemented
from header import CWD

from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from os import listdir
from os.path import join
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv


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

def saveHistory(path, history):
    df_history = DataFrame(history.history)
    hist_csv_file = join(path, 'hist_'+getTimeStamp()+'.csv')
    with open(hist_csv_file, mode='w') as f: df_history.to_csv(f)

def loadHistory(path):
    return read_csv(path)

def plotResult(argument: str, result: DataFrame, validation: DataFrame, savePath: str):
    fig = plt.figure(figsize=(10,6))
    plt.plot(result.to_list(), label='Train')
    plt.plot(validation.to_list(), label='Validation')
    plt.title(argument); plt.xlabel('Epoch'); plt.ylabel(argument); plt.legend(); plt.tight_layout()
    plt.savefig(join(savePath, argument+'.jpg'))
    return fig

def pltPRCurve(prec: DataFrame, rec: DataFrame, precVal: DataFrame, recVal: DataFrame, savePath: str):
    fig = plt.figure(figsize=(10,6))
    plt.plot(rec.to_list(), prec.to_list(), label='Precision-Recall Curve')
    plt.plot(recVal.to_list(), precVal.to_list(), label='Precision-Recall Validation Curve')
    plt.title('Precision-Recall Curve'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.tight_layout()
    plt.savefig(join(savePath, 'PrecRecCurve.jpg'))
    return fig

def calculateIoU(tp: DataFrame, fp: DataFrame, fn: DataFrame):
    return (tp/(tp+fp+fn))

def calculatePrec(tp: DataFrame, fp: DataFrame):
    return (tp/(tp+fp))

def calculateRec(tp: DataFrame, fn: DataFrame):
    return (tp/(tp+fn))

def calculateF1(prec: DataFrame, rec: DataFrame):
    return ((2*prec*rec)/(prec+rec))

def evalModelTraining(savePath):
    historyPath = [join(savePath, file) for file in listdir(savePath) if file.endswith('.csv')]
    df = loadHistory(historyPath[0])

    tp=df['true_positives']; tn=df['true_negatives']; fp=df['false_positives']; fn=df['false_negatives']
    tpVal=df['val_true_positives']; tnVal=df['val_true_negatives']; fpVal=df['val_false_positives']; fnVal=df['val_false_negatives']
    prec = calculatePrec(tp=tp, fp=fp)
    precVal = calculatePrec(tp=tpVal, fp=fpVal)
    rec = calculateRec(tp=tp, fn=fn)
    recVal = calculateRec(tp=tpVal, fn=fnVal)

    iou = calculateIoU(tp=tp,fp=fp,fn=fn)
    iouVal = calculateIoU(tp=tpVal, fp=fpVal, fn=fnVal)
    f1 = calculateF1(prec=prec, rec=rec)
    f1Val = calculateF1(prec=precVal, rec=recVal)

    # Loss Evaluation
    lossFig = plotResult('Loss',df['loss'], df['val_loss'],savePath=savePath)
    # Accuracy Evaluation
    accFig = plotResult('Accuracy', df['accuracy'], df['val_accuracy'], savePath=savePath)
    # IoU Evaluation
    iouFig = plotResult('Intersection over Union (IoU)',iou, iouVal, savePath=savePath)
    # F1 Evaluation
    f1Fig = plotResult('F1-Score', f1, f1Val, savePath=savePath)
    # Preciscion-Recall Curve
    prCurve = pltPRCurve(prec=prec, precVal=precVal, rec=rec, recVal=recVal, savePath=savePath)