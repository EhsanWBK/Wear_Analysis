''' ## Functions for Training Evaluation

Metrics: accuracy, loss; iou score

Accessible Functions:
- evalTraining()
'''
from header import METRICS
from numpy import logical_and, logical_or, sum, squeeze
import matplotlib.pyplot as plt

def iouScore(yPred, yTruth):
    '''Evaluation of the accuracy of the predctions with Intersection over Union Score (IoU).'''
    yPred = squeeze(yPred, axis=-1)
    yPredThresh = yPred > 0.5 # apply threshold to the prediction
    intersect = logical_and(yTruth, yPredThresh)
    union = logical_or(yTruth, yPredThresh)
    iou = sum(intersect)/sum(union)
    print('\nIoU Score:\t\t', iou)
    return iou

def prCurve():
    '''Evaluation of Training with Precision-Recall Curve.'''
    return True

def learningCurve(history, metric = 'loss'):
    '''
    Evaluation of Training by plotting Learning Curve (pyplot). 
    - history: (saved) history of model training (monitoring of training)
    - metric: 'loss', 'accuracy'
    '''
    metrics = history.history[metric]
    valMetrics = history.history['val_'+str(metric)]
    epochs = range(1, len(metrics)+1)

    plt.plot(epochs, metrics, 'y', label='Training '+str(metric))
    plt.plot(epochs, valMetrics, 'r', label='Validation '+str(metric))
    plt.title('Training and validataion '+str(metric))
    plt.xlabel('epochs')
    plt.ylabel(str(metric))
    plt.legend
    plt.show()
    return True

def displayScores():
    ''' Function to display the scores of the training in terminal.'''
    return True

def evalTraining(modelTrained, history, testData: dict):
    '''
    Evaluating the Model Training based on:
    - IoU Score
    - Learning Curve (loss & accuracy)
    - Precision-Recall Curve

    Takes in model, model history, and data to test on. Returns scores.
    Plots the scores too.
    '''
    testPred = modelTrained.predict(testData['xTest'])
    # evaluations
    iou = iouScore(testPred, testData['yTest'])
    learningCurve(history=history, metric=METRICS[0])
    learningCurve(history=history, metric=METRICS[1])
    prCurve()
    return iou
