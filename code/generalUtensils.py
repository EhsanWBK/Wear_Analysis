from numpy import ndarray, expand_dims, array
from os import listdir, remove, mkdir, getcwd
from os.path import join, isfile, isdir, splitext, exists
from tqdm import tqdm
from cv2 import imread, imwrite, imencode, resize, INTER_LINEAR
from base64 import b64encode
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.models import load_model, Model

from header import MODEL_FORMAT, PNG_SUFFIX, TIF_SUFFIX

#  =========================================
#  	           General Functions		
#  =========================================

def pathCreator(projectPath: str, grabData: bool=False, token: str='img') -> str:
    '''Create paths to image and mask folder. Takes in path to project.'''
    imgPath = join(projectPath, 'images'); maskPath = join(projectPath, 'masks')
    if grabData: imgPath = join(imgPath, token); maskPath = join(maskPath, token)
    return imgPath, maskPath

def getTimeStamp() -> datetime:
    return datetime.now().strftime('%Y_%d%m_%H%M%S')

def reformatFrame(frame):
    ''' Reformats image and converts to utf-8 BLOB (binary large object).'''
    frameResize = resize(frame, (512, 512), interpolation=INTER_LINEAR)
    jpeg = imencode('.jpeg', frameResize)[1]
    byte_data = jpeg.tobytes() #
    blob = b64encode(byte_data)
    return blob.decode("utf-8")

#  =========================================
#  	         Directory Functions		
#  =========================================

def imageReader(targetPath: str, segment: bool = False) -> ndarray:
    ''' Read images from path to image file or directory. Returns numpy array with image data.
    Path can either specific file or project name. Function reads image from directory.
    The option "segment" tells, if expanded dimensions for the segementation are needed.
    '''
    fileNames = []; data=[]
    if isfile(path=str(targetPath)): # if path is file, read and return image
        img = imread(targetPath, 0)
        img = expand_dims(img, 2)
        img = img[:,:,0][:,:,None]
        img = expand_dims(img,0)     
        return img, None
    elif isdir(s=str(targetPath)): # if path is directory, read and return stack of images
        for img_name in listdir(path=targetPath):
            name = splitext(img_name)[0]
            fileNames.append(name)
            PATH = join(targetPath, img_name)
            if isfile(PATH):
                img = imread(PATH,0)
                img = expand_dims(img, 2) if segment else img
                data.append(img)
        data = array(data, dtype='float')
        return data, fileNames

def setupFolder(folderPath: str, token: str = None) -> None:
    ''' Sets up folder inside the selected project. 
    "Token" is the name of the name of the folder. E.g. token = "resized" for resized images.
    Path is the path to the directory where the new folder is created.
    In the process, the function also deletes all currently exisiting folders with the same name. '''
    path_dir = join(folderPath, token)
    if isdir(path_dir):
        print('\nSETUP FOLDER:\nClearing folder:\t', path_dir)
        try: 
            files = listdir(path_dir)
            for file in files: file_path = join(path_dir, file); remove(file_path)
        except Exception as e: print(f"Error deleting files: {e}") 
    else:
        try: mkdir(path=path_dir, mode=777)
        except Exception as e:  print(f'Error creating new folder: {e}')
        
def saveFrame(pathTarget: str, image: list, token: str, names=['test'], maskConversion: bool=False, suffix: str=PNG_SUFFIX) -> None:
    ''' Target path is either image or mask path. Also the image data and the folder name
    under which the images are saved in. The folder will be created from scratch.'''
    setupFolder(folderPath=pathTarget, token=token)
    for imgEntry in range(len(image)):
        filename=join(pathTarget, token, token+'_'+str(names[imgEntry])+suffix)
        imwrite(filename, image[imgEntry])
    if maskConversion:  convertMaskFileType(projectPath=pathTarget, token=token)

def saveTrigger(triggerList: list):
    triggerPath = join(getcwd(), 'results','trigger',getTimeStamp())
    if not exists(path=triggerPath): mkdir(path=triggerPath)
    for idx in range(len(triggerList)):
        filename = join(triggerPath, 'trigger_save_'+str(idx)+'_'+getTimeStamp()+PNG_SUFFIX)
        imwrite(filename, triggerList[idx])

def convertMaskFileType(projectPath, token):
    sourceDirectory = join(projectPath, token)
    setupFolder(folderPath=projectPath, token=str(token+'_tif'))
    targetDirectroy = join(projectPath, token+'_tif')
    suffix = TIF_SUFFIX
    for fileName in tqdm(listdir(path=sourceDirectory)):
        if fileName.endswith(('.jpg','.jpeg','.png')):
            maskPath = join(sourceDirectory, fileName)
            mask = Image.open(maskPath)
            targetPath=join(targetDirectroy, splitext(fileName)[0]+suffix)
            mask.save(targetPath, 'TIFF')

def loadCurModel(path: str) -> Model:
    model = load_model(path); print(model.summary())
    return model

def saveCurModel(model: Model, modelPath: str) -> bool:
    timeStamp = getTimeStamp() # name model according to time stamp
    dirPath = join(modelPath,'Model_Training_'+timeStamp+MODEL_FORMAT)
    print('\nSaving Model to:\n'+dirPath+'\n')
    model.save(dirPath)
    return True

#  =========================================
#  	            Data Functions		
#  =========================================

def displayImg(imgInput: ndarray) -> None:
    fig = plt.figure(figsize=(12, 10))
    for img in range(1, 13, +1):
        ax = fig.add_subplot(3, 4, img)
        plt.imshow(imgInput[img], interpolation='nearest', cmap='gray')
    plt.show()

def setupData(projectPath: str, par: dict = None, split: bool = True, token: str = 'img') -> dict:
    ''' Takes in project path and returns data. If "split" is True, data is split into training and testing data set. Data is stored in dictionary with keywords:
    'xTrain', 'yTrain', 'xTest', 'yTest'. If False returns image and mask data instead. '''
    imgPath, maskPath = pathCreator(projectPath=projectPath, grabData=True, token=token)
    if token == 'final': maskPath= join(projectPath, 'masks', 'final_tif')
    images, imageNames = imageReader(imgPath)
    if exists(maskPath): masks, maskNames = imageReader(maskPath)
    else: masks = None; maskNames = None
    if split:
        xTrain, xTest, yTrain, yTest = train_test_split(images, masks, test_size=float(par['validationSize']), random_state=int(par['randomState']), shuffle=bool(par['randomSelection']))
        return {'xTrain': xTrain, 'yTrain': yTrain, 'xTest': xTest,'yTest': yTest}
    else: return images, masks, imageNames, maskNames

def getAugFilePaths(pathImg, pathMask, extension):
    imgPaths = [join(pathImg, f) for f in listdir(pathImg) if isfile(join(pathImg, f)) and f.endswith(extension)]
    maskPaths = [join(pathMask, f) for f in listdir(pathMask) if isfile(join(pathMask, f)) and f.endswith(extension)]
    imgNames = [name for name in listdir(path=pathImg)]; maskNames = [name for name in listdir(path=pathMask)]
    return imgPaths, maskPaths, imgNames, maskNames

def setupAugemented(projectPath: str, parDic: dict = None, split: bool = True):
    horFlip = bool(parDic['horizontalFlip']); verFlip = bool(parDic['verticalFlip']); rot90deg = bool(parDic['rot90deg'])
    imgPath = join(projectPath, 'images', 'final'); maskPath = join(projectPath, 'masks', 'final_tif')
    trainImgData = []; trainMaskData = []; trainImgNames = []; trainMaskNames = []
    augmentations = [(horFlip, '_flip_hor'), (verFlip, '_flip_ver'), (rot90deg, '_rot_90')]
    for flag, suffix  in augmentations: 
        if flag: 
            augImgPath, augMaskPath, augImgNames, augMaskNames = getAugFilePaths(pathImg=imgPath, pathMask=maskPath, extension='')
            augImg = [imread(filePath, 0) for filePath in augImgPath]; augMask = [imread(filePath, 0) for filePath in augMaskPath]
            trainImgData.extend(augImg); trainMaskData.extend(augMask); trainImgNames.extend(augImgNames); trainMaskNames.extend(augMaskNames)
    trainImgData = array(trainImgData, dtype='float'); trainMaskData = array(trainMaskData, dtype='float')
    if split:
        xTrain, xTest, yTrain, yTest = train_test_split(trainImgData, trainMaskData, test_size=float(parDic['validationSize']), random_state=int(parDic['randomState']), shuffle=bool(parDic['randomSelection']))
        return {'xTrain': xTrain, 'yTrain': yTrain, 'xTest': xTest,'yTest': yTest}
    else: return None
