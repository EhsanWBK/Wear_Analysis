from numpy import ndarray, unique, linspace, expand_dims, array, squeeze, rot90, eye, float32
from cv2 import resize, INTER_LINEAR,  flip, MOTION_TRANSLATION, TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT, INTER_LINEAR, WARP_INVERSE_MAP, findTransformECC, warpAffine
from copy import deepcopy
from os.path import exists
from os import makedirs
from tensorflow import config as cfg

from generalUtensils import setupData, saveFrame, pathCreator

#  =========================================
#  	     Single Pre-Processing Steps		
#  =========================================

def cropImage(imageData: ndarray, deltaX: int = 700, deltaY: int = 1000) -> list:
    croppedImages = []; imgheight=imageData.shape[1]; imgwidth=imageData.shape[2]
    for entry in imageData: # pass one or multiple images  
        start_x = max(0, int(imgwidth/2) - deltaX)
        end_x = min(imgwidth, int(imgwidth/2) + deltaX)
        start_y = max(0, int(imgheight/2) - deltaY)
        end_y = min(imgheight, int(imgheight/2) + deltaY)
        cropped_image = entry[start_y:end_y, start_x:end_x] # slicing image
        croppedImages.append(cropped_image)
    return croppedImages

def alignImage(imageData: ndarray) -> ndarray:
    WARP_MODE = MOTION_TRANSLATION
    WARP_MATRIX = eye(2, 3, dtype=float32)
    NR_ITERATIONS = 10000
    TERMINATOR = 1e-10
    CRITERIA = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, NR_ITERATIONS, TERMINATOR) 
    SZ = imageData[0].shape
    imageData = imageData.astype('float32')
    REF_IMG = imageData[0]
    try:
        for img in range(len(imageData)-1): 
            (_, WARP_MATRIX) = findTransformECC(REF_IMG,imageData[img+1],WARP_MATRIX, WARP_MODE, CRITERIA) 
            aligned_img = warpAffine(imageData[img], WARP_MATRIX, (SZ[1],SZ[0]), flags=INTER_LINEAR + WARP_INVERSE_MAP) 
    except:print('Warning: find transform failed.'); return None, False
    return aligned_img, True

def resizeSingleFrame(frame: ndarray, aspectRatio = None, channel: int = None) -> ndarray:
    ''' Resize single frame according to parameters. Returns resizedImages, resizedMasks, token.
    Default aspect ration 512x512 to match current model input. '''
    aspectRatio = (512,512) if aspectRatio is None else aspectRatio
    if len(frame.shape) == 4: frame = squeeze(frame)
    return resize(frame, aspectRatio, interpolation=INTER_LINEAR)

#  =========================================
#  	         Pre-Processing Steps		
#  =========================================

def alignAll(img: ndarray, projectPath: str, token: str='aligned', saveProgress: bool=True) -> ndarray:
    imgAligned, imgSuccess = alignImage(imageData=img)
    if not (imgSuccess):print('Error occured in alignment: saving original images'); imgAligned = img
    imgPath, _ = pathCreator(projectPath=projectPath)
    if saveProgress: saveFrame(pathTarget=imgPath, image=imgAligned, token=token, imgData=False)
    return imgAligned, token

def augementAll(img: ndarray, mask: ndarray, projectPath: str, fileNames: str, token: str='aug', saveProgress: bool=True):
    cfg.run_functions_eagerly(True)
    imgPath,maskPath = pathCreator(projectPath=projectPath)
    if not exists(path=imgPath): makedirs(imgPath)
    if not exists(path=maskPath): makedirs(maskPath)
    imgList = []; maskList = []; imgNamesList = []; maskNamesList = []
    augmentations = [("_original", lambda x: x), ("_flip_hor", lambda x: flip(x, 1)), ("_flip_ver", lambda x: flip(x, 0)), ("_rot_90", lambda x: rot90(x))]
    for i, (img, mask, imgName, maskName) in enumerate(zip(img, mask, fileNames[0], fileNames[1])):
        for suffix, augmentation in augmentations:
            imgList.append(augmentation(img)); maskList.append(augmentation(mask))
            imgNamesList.append(f'{img}{suffix}'); maskNamesList.append(f'{mask}{suffix}')
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

def resizeAll(img: ndarray, fileNames: str = '', projectPath: str = '', aspectRatio: tuple = None, mask: ndarray = [], token: str = 'resized', saveProgress: bool=True) -> ndarray:
    imagesResized = []; maskResized = []
    aspectRatio = (512,512) if aspectRatio is None else aspectRatio
    if projectPath is not '': imgPath, maskPath = pathCreator(projectPath=projectPath)
    for singleFrame in img: resizedFrame = resizeSingleFrame(frame=singleFrame, aspectRatio=aspectRatio); imagesResized.append(resizedFrame)
    if saveProgress: saveFrame(pathTarget=imgPath, image=imagesResized, names=fileNames[0], token=token, maskConversion=True)
    if mask is not None:
        for singleFrame in mask: resizedFrame = resizeSingleFrame(frame=singleFrame, aspectRatio=aspectRatio); maskResized.append(resizedFrame)
        if saveProgress: saveFrame(pathTarget=imgPath, image=maskResized, names=fileNames[1], token=token, maskConversion=True)
    return array(imagesResized), array(maskResized), token

def maskConversion(masks: ndarray, numClasses: int=2) -> ndarray:
    '''Correct masks where the number of colors does not correspond to the number of classes (due to resizing or data compression).
    Uses linear distributed thresholds. '''
    uniqueColors = set() # find unique colors
    for img in masks: uniqueColors |= set(unique(masks[img]))
    minimum = min(uniqueColors); maximum = max(uniqueColors)
    threshold = linspace(start=minimum,stop=maximum+1,num=numClasses+1)
    colorClasses = linspace(start=0, stop=255, num=numClasses)
    for i in range(len(masks)):
        for c in range(numClasses):
            masks[(threshold[c]<= masks)&(threshold[c+1]>masks)] = colorClasses[c]
    return masks

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

def preProcForSegment(imgArray: ndarray, projectPath: str, fileNames:list, aspectRatio = (512,512), channel: int = 1) -> None:
    ''' Takes in an array from the data storage and saves a subfolder 'segmentation' with pre-processed images.'''
    imgPath, maskPath = pathCreator(projectPath=projectPath); segmentImg = []
    for img in imgArray:
        upscaledFrame = resize(img, (2048,2448), interpolation=INTER_LINEAR)
        segmentFrame = preProcFromCamera(frame=upscaledFrame, aspectRatio=aspectRatio, channel=channel)
        segmentImg.append(segmentFrame)
    saveFrame(pathTarget=imgPath, image=segmentImg, fileNames=fileNames[0], token='seg')

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
    else: return imgArray
    
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

def preProcFromCamera(frame: ndarray, aspectRatio: tuple = (512,512), channel: int = 1) -> ndarray:
    ''' Takes in an array of shape (2048,2448,3) from the camera live stream. 
    Returns (1,imWidth, imHeight, channel) array for segmentation.
    Applies for single images only. For multiple images, iterate over the function. '''
    frameCrop = cropImage(frame)
    frameResize = resizeSingleFrame(frame=frameCrop, aspectRatio=aspectRatio, channel=channel)
    return frameResize