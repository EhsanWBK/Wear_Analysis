''' ## General Utenesils and useful functions

Accessible Functions:
- imageReader()
- saveImage()
- displayImage()
- setupData()
'''

from numpy import ndarray, expand_dims, array
import numpy
import os
from tqdm import tqdm
from cv2 import imread, imwrite
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
# from tensorflow import keras
import tensorflow as tf

from header import CWD, IMG_TARGET_SUFFIX, MASK_TARGET_SUFFIX, TEST_SIZE, SPLIT_RANDOM_STATE

def imageReader(targetPath: str, segment: bool = False) -> ndarray:
    ''' Read images from path to image file or directory. Returns numpy array with image data.
    Path can either specific file or project name. Function reads image from directory.
    The option "segment" tells, if expanded dimensions for the segementation are needed.
    '''
    print('\nReading Image(s) for:\t',targetPath)
    if os.path.isfile(path=str(targetPath)): # if path is file, read and return image
        print('Path is file.\n')
        img = imread(targetPath, 0)
        # img = expand_dims(normalize(array(img), axis=1), 2)
        img = expand_dims(img, 2)
        img = img[:,:,0][:,:,None]
        img = expand_dims(img,0)     
        return img
    elif os.path.isdir(s=str(targetPath)): # if path is directory, read and return stack of images
        print('Path is Directory.\n')
        data=[]
        for img_name in tqdm(os.listdir(path=targetPath)):
            PATH = os.path.join(targetPath, img_name)
            if os.path.isfile(PATH):
                img = imread(PATH,0)
                img = expand_dims(img, 2) if segment else img
                data.append(img)
        data = array(data, dtype='float')
        return data

def setupFolder(folderPath: str, token: str = None) -> None:
    ''' Sets up folder inside the selected project. 
    "Token" is the name of the name of the folder. E.g. token = "resized" for resized images.
    Path is the path to the directory where the new folder is created.
    In the process, the function also deletes all currently exisiting folders with the same name.
    '''
    path_dir = os.path.join(folderPath, token)
    if os.path.isdir(path_dir):
        print('Clearing folder:\t', path_dir)
        try: 
            files = os.listdir(path_dir)
            for file in files:
                file_path = os.path.join(path_dir, file)
                os.remove(file_path)
            print('Removed all contents')
        except Exception as e:
            print(f"Error deleting files: {e}") 
    else:
        try:
            os.mkdir(path=path_dir, mode=777)
        except Exception as e:
            print(f'Error creating new folder: {e}')
            return
        
def saveImage(pathTarget: str, image: Image, token: str, imgData: bool = True) -> None:
    ''' Input is the path to the project the images are saved to. Also the image data and the folder name
    under which the images are saved in. The folder will be created from scratch.
    The type of images is by default "Image". If masks are saved, imgData needs to be False.
    Unspecified images will be saved to the 'saved_images' folder. '''

    folderPath = os.path.join(CWD, pathTarget) # path to the folder to store the images in
    # setupFolder(folderPath=folderPath, token=token) # set up folder before storing
    if imgData: # or pathTarget[-6:] == 'images'
        suffix = IMG_TARGET_SUFFIX
        subfolder = 'images'
    elif not imgData: # or pathTarget[-5:] == 'masks'
        suffix = MASK_TARGET_SUFFIX
        subfolder = 'masks'
    else:
        folderPath = os.path.join(CWD, 'saved_images')
        suffix = IMG_TARGET_SUFFIX
    print('\nFolder Path: ', folderPath)
    subfolderPath = os.path.join(folderPath, subfolder) # create subfolder path
    setupFolder(folderPath=subfolderPath, token=token)
    for imgEntry in range(len(image)):
        imgName = str(imgEntry)
        filename=os.path.join(folderPath, subfolder, token, token+imgName+suffix)
        imwrite(filename, image[imgEntry])

def displayImg(imgInput: ndarray) -> None:
    ''' Plots images from stack of numpy image data.'''
    fig = plt.figure(figsize=(12, 10))
    for img in range(1, 13, +1):
        ax = fig.add_subplot(3, 4, img)
        plt.imshow(imgInput[img], interpolation='nearest', cmap='gray')
    plt.show()

def pathCreator(projectPath: str) -> str:
    '''Create paths to image and mask folder. Takes in path to project.
    UNUSED'''
    imgPath = os.path.join(projectPath, 'images', 'img')
    maskPath = os.path.join(projectPath, 'masks', 'img')
    return imgPath, maskPath


def setupData(projectPath: str, split: bool = True) -> dict:
    ''' Takes in project path and returns data. 

    If "split" is True, data is split into training and testing data set. Data is stored in dictionary with keywords:
    - 'xTrain'
    - 'yTrain'
    - 'xTest'
    - 'yTest'

    If False returns image and mask data instead.
    '''
    imgPath, maskPath = pathCreator(projectPath=projectPath)
    images = imageReader(imgPath)
    masks = imageReader(maskPath)
    # train-test-split
    if split:
        xTrain, xTest, yTrain, yTest = train_test_split(images, masks, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
        data = {
            'xTrain': xTrain,
            'yTrain': yTrain,
            'xTest': xTest,
            'yTest': yTest
        }
        return data
    else: return images, masks
    
def getTestData(trainData:dict) -> dict:
    ''' Extracts test data from data dictionary. '''
    testData = dict((k, trainData[k]) for k in ('xTest', 'yTest'))
    return testData
    
def getTimeStamp() -> datetime:
    ''' Returns current time stamp.'''
    now = datetime.now()
    timeStamp = now.strftime('%Y_%d%m_%H%M')
    return timeStamp

def loadCurModel(path: str) -> tf.keras.models.Model:
    ''' Loads model according to path to the saved model file.'''
    model = tf.keras.models.load_model(path)
    print(model.summary())
    return model