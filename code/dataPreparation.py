'''## Offline Data Preprocessing

Processing raw image data by creating specific folders for the image manipulation task.
Takes in 'Project Name' the images are filed for.

Accessible Functions:
- offlinePreProc()
- alignAll(img: ndarray, mask: ndarray, token=None)
- augmentAll(img: ndarray, mask: ndarray, token=None)
- cropAll(img: ndarray, mask: ndarray, token=None)
- convertAll(img: ndarray, mask: ndarray, token=None)
- resizeImage(parameters: dict, data: ndarray)
- maskConversion(masks, numClasses)
'''

from numpy import ndarray, unique, linspace, expand_dims, array, squeeze, rot90
from cv2 import resize, INTER_LINEAR, imwrite, flip
from copy import deepcopy
from os.path import join, exists
from os import makedirs
from tensorflow import config as cfg
from tensorflow import image as tfImg

from generalUtensils import setupData, saveFrame, pathCreator
from crop_align import alignImage, cropImage

#  =========================================
#  	     Single Pre-Processing Steps		
#  =========================================

def resizeSingleFrame(frame: ndarray, aspectRatio = None, channel: int = None) -> ndarray:
    ''' Resize single frame according to parameters. Returns resizedImages, resizedMasks, token.
    Default aspect ration 512x512 to match current model input. '''
    aspectRatio = (512,512) if aspectRatio is None else aspectRatio
    if len(frame.shape) == 4: frame = squeeze(frame)
    resizedFrame = resize(frame, aspectRatio, interpolation=INTER_LINEAR)
    return resizedFrame

#  =========================================
#  	         Pre-Processing Steps		
#  =========================================

def alignAll(img: ndarray, projectPath: str, token: str='aligned', saveProgress: bool=True) -> ndarray:
    ''' Aligning stack of images and masks. Only used for image segmentation of data stack. '''
    imgAligned, imgSuccess = alignImage(imageData=img)
    if not (imgSuccess):
        print('Error occured in alignment: saving original images')
        imgAligned = img
    imgPath, _ = pathCreator(projectPath=projectPath)
    if saveProgress: saveFrame(pathTarget=imgPath, image=imgAligned, token=token, imgData=False)
    return imgAligned, token

def augementAll(img: ndarray, mask: ndarray, projectPath: str, fileNames: str, token: str='aug', saveProgress: bool=True):
    ''' Enhancing training by increased number of images. Used in pre-processing for training images. '''
    cfg.run_functions_eagerly(True)
    imgPath,maskPath = pathCreator(projectPath=projectPath)
    if not exists(path=imgPath): makedirs(imgPath)
    if not exists(path=maskPath): makedirs(maskPath)
    imgList = []
    maskList = []
    imgNamesList = []
    maskNamesList = []

    for i in range(len(img)):
        imgList.append(img[i])
        imgList.append(flip(deepcopy(img[i]), 1))
        imgList.append(flip(deepcopy(img[i]),0))
        imgList.append(rot90(img[i]))
        maskList.append(mask[i])
        maskList.append(flip(deepcopy(mask[i]), 1))
        maskList.append(flip(deepcopy(mask[i]),0))
        maskList.append(rot90(mask[i]))

        imgNamesList.append(str(fileNames[0][i])+'_original')
        imgNamesList.extend([str(fileNames[0][i])+'_flip_hor', str(fileNames[0][i])+'_flip_ver', str(fileNames[0][i])+'_rot_90'])
        maskNamesList.append(str(fileNames[1][i])+'_original')
        maskNamesList.extend([str(fileNames[1][i])+'_flip_hor', str(fileNames[1][i])+'_flip_ver', str(fileNames[1][i])+'_rot_90'])
        
    if saveProgress:
        saveFrame(image=imgList, pathTarget=imgPath, names=imgNamesList, token=token)
        saveFrame(image=maskList, pathTarget=maskPath, names=maskNamesList, token=token, maskConversion=True)
    return imgList, maskList, token, [imgNamesList, maskNamesList]

def cropAll(img: ndarray, mask: ndarray, projectPath: str, fileNames: list, token: str='cropped', saveProgress: bool=True) -> ndarray:
    ''' Crop stack of images and masks. '''
    img = cropImage(imageData=array(img))
    try: mask = cropImage(imageData=array(mask)) 
    except: pass

    imgPath, maskPath = pathCreator(projectPath)
    if saveProgress:
        saveFrame(pathTarget=imgPath, image=img, token=token, names=fileNames[0])
        try: saveFrame(pathTarget=maskPath, image=mask, token=token, names=fileNames[1], maskConversion=True)
        except: pass
    return img, mask, token

def convertAll(img: ndarray, mask: ndarray, projectPath: str, fileNames: list, token: str='converted', saveProgress: bool=True) -> ndarray:
    imgPath, maskPath = pathCreator(projectPath)
    if saveProgress:
        saveFrame(pathTarget=imgPath, image=img, token=token, names=fileNames[0])
        try:saveFrame(pathTarget=maskPath, image=mask, token=token, names=fileNames[1], maskConversion=True)
        except: pass
    return img, mask, token

def resizeAll(img: ndarray, fileNames: str, projectPath: str = '', aspectRatio: tuple = None, mask: ndarray = [], token: str = 'resized', saveProgress: bool=True) -> ndarray:
    imagesResized = []
    maskResized = []
    aspectRatio = (512,512) if aspectRatio is None else aspectRatio
    if projectPath is not '': imgPath, maskPath = pathCreator(projectPath=projectPath)
    for singleFrame in img:
        resizedFrame = resizeSingleFrame(frame=singleFrame, aspectRatio=aspectRatio)
        imagesResized.append(resizedFrame)
    if saveProgress: saveFrame(pathTarget=imgPath, image=imagesResized, names=fileNames[0], token=token, maskConversion=True)
    if mask is not None:
        print('Mask is not None.')
        for singleFrame in mask:
            resizedFrame = resizeSingleFrame(frame=singleFrame, aspectRatio=aspectRatio)
            maskResized.append(resizedFrame)
        if saveProgress: saveFrame(pathTarget=imgPath, image=maskResized, names=fileNames[1], token=token, maskConversion=True)
    return array(imagesResized), array(maskResized), token


#  =========================================
#  	        Offline Pre-Processing		
#  =========================================

def preProcStart(argument, projectPath, aspectRatio):
    img, mask, imgNames, maskNames = setupData(projectPath=projectPath, split=False)
    fileNames = [imgNames, maskNames]
    if argument[0] == 'augment': augementAll(img=img, mask=mask, fileNames=fileNames, projectPath=projectPath) # does not make any sense
    elif argument[0] == 'crop': cropAll(img=img, mask=mask, fileNames=fileNames, projectPath=projectPath)
    elif argument[0] == 'convert': convertAll(img=img, mask=mask, fileNames=fileNames, projectPath=projectPath)
    elif argument[0] == 'resize': resizeAll(img=img, mask=mask, projectPath=projectPath, fileNames=fileNames, aspectRatio=aspectRatio)
    elif argument[0] == 'training': preProcForDataStorage(frame=img, mask=mask, projectPath=projectPath, fileNames=fileNames, aspectRatio=aspectRatio)
    elif argument[0] == 'segment': preProcForSegment(imgArray=img, projectPath=projectPath, fileNames=fileNames, aspectRatio=aspectRatio)
    else: print('Unknown argument')
    print('\nFinished Pre-Processing')

def preProcFromCamera(frame: ndarray, aspectRatio: tuple = (512,512), channel: int = 1) -> ndarray:
    ''' Takes in an array of shape (2048,2448,3) from the camera live stream. 
    Returns (1,imWidth, imHeight, channel) array for segmentation.
    Applies for single images only. For multiple images, iterate over the function. '''
    frameCrop = cropImage(frame)
    frameResize = resizeSingleFrame(frame=frameCrop, aspectRatio=aspectRatio, channel=channel)
    return frameResize

def preProcForSegment(imgArray: ndarray, projectPath: str, fileNames:list, aspectRatio = (512,512), channel: int = 1) -> None:
    ''' Takes in an array from the data storage and saves a subfolder 'segmentation' with pre-processed images.'''
    imgPath, maskPath = pathCreator(projectPath=projectPath)
    segmentImg = []
    for img in imgArray:
        upscaledFrame = resize(img, (2048,2448), interpolation=INTER_LINEAR)
        segmentFrame = preProcFromCamera(frame=upscaledFrame, aspectRatio=aspectRatio, channel=channel)
        segmentImg.append(segmentFrame)
    saveFrame(pathTarget=imgPath, image=segmentImg, fileNames=fileNames[0], token='seg')
    print('Saved all images')


def preProcFromDataStorage(imgArray: ndarray, saveProgress: bool = False, segment: bool = True):
    ''' Takes in an array from the data storage and returns either a (1,imWidth, imHeight, channel) array for segmentation
    or a (imWidth,imHeight) array for model training. '''
    if segment:
        print('Images Passed for Pre Processing: ',len(imgArray))
        if len(imgArray) > 1: imgTemp, _ = alignAll(img=imgArray, saveProgress=saveProgress)
        else: imgTemp = imgArray
        imgExpand = expand_dims(imgTemp, 2)
        imgExpand = imgExpand[:,:,0][:,:,None]
        imgExpand = expand_dims(imgTemp, 0)
        return imgExpand
    else:
        return imgArray
    
def preProcForDataStorage(frame: ndarray, mask: ndarray, projectPath: str, fileNames: list, aspectRatio = (512,512)) -> None:
    ''' Takes in high-resolution image frame and mask of shape (2048,2448,3) and (2048,2448) from 'img' subfolder.
    Saves images to subfolders of the project path after 1) cropping, 2) resizing.
    Resizing to target aspect ratio. Saves images and masks to pre-defined file formats in 'final' folder. '''
    print('\nCROPPING IMAGES')
    frameCrop, maskCrop, _ = cropAll(img=frame, mask=mask, projectPath=projectPath, fileNames=fileNames)
    print('\nRESIZING IMAGES')
    frameResize, maskResize, _ = resizeAll(img=frameCrop, mask=maskCrop, projectPath=projectPath, fileNames=fileNames, aspectRatio=aspectRatio)
    print('\nAUGMENTING IMAGES')
    frameAug, maskAug, _, augNames = augementAll(img=frameResize, mask=maskResize, projectPath=projectPath, fileNames=fileNames)
    print('\nCONVERTING IMAGES')
    frameFinal, maskFinal, _ = convertAll(img=frameAug, mask=maskAug, projectPath=projectPath, fileNames=augNames, token='final')

#  =========================================
#  	        Online Pre-Processing		
#  =========================================

# def onlinePreProc(frame):
#     frame = cvtColor(frame, COLOR_BGR2GRAY)
#     frame = resize(frame, (512,512))
#     img = expand_dims(frame, 2)
#     img = img[:,:,0][:,:,None]
#     img = expand_dims(img,0)
#     return img

#  =========================================
#  	           Unused Functions		
#  =========================================

def maskConversion(masks: ndarray, numClasses: int=2) -> ndarray:
    '''Correct masks where the number of colors does not correspond to the number of classes (due to resizing or data compression).
    Uses linear distributed thresholds. '''
    uniqueColors = set() # find unique colors
    for img in masks:
        uniqueColors |= set(unique(masks[img]))
    minimum = min(uniqueColors)
    maximum = max(uniqueColors)
    threshold = linspace(start=minimum,stop=maximum+1,num=numClasses+1)
    colorClasses = linspace(start=0, stop=255, num=numClasses)
    for i in range(len(masks)):
        for c in range(numClasses):
            masks[(threshold[c]<= masks)&(threshold[c+1]>masks)] = colorClasses[c]
    return masks