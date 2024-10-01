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

from numpy import ndarray, unique, linspace, expand_dims, array, squeeze
from cv2 import resize, INTER_LINEAR

from generalUtensils import setupData, saveFrame, pathCreator
from crop_align import alignImage, cropImage

def resizeSingleFrame(frame: ndarray, aspectRatio = None, channel: int = None) -> ndarray:
    ''' Resize single frame according to parameters. Returns resizedImages, resizedMasks, token.
    Default aspect ration 512x512 to match current model input. '''
    aspectRatio = (512,512) if aspectRatio is None else aspectRatio
    if len(frame.shape) == 4: frame = squeeze(frame)
    resizedFrame = resize(frame, aspectRatio, interpolation=INTER_LINEAR)
    return resizedFrame

#  =========================================
#  	        Pre-Processing Steps		
#  =========================================

def alignAll(img: ndarray, projectPath: str, token: str='aligned', saveProgress: bool=True) -> ndarray:
    ''' Aligning stack of images and masks. '''
    imgAligned, imgSuccess = alignImage(imageData=img)

    if not (imgSuccess):
        print('Error occured in alignment: saving original images')
        imgAligned = img

    imgPath, _ = pathCreator(projectPath=projectPath)
    if saveProgress:
        saveFrame(pathTarget=imgPath, image=imgAligned, token=token, imgData=False)
    return imgAligned, token

def cropAll(img: ndarray, mask: ndarray, projectPath: str, token: str='cropped', saveProgress: bool=True) -> ndarray:
    ''' Crop stack of images and masks. '''
    img = cropImage(imageData=array(img))
    mask = cropImage(imageData=array(mask))

    imgPath, maskPath = pathCreator(projectPath)
    if saveProgress:
        saveFrame(pathTarget=imgPath, image=img, token=token)
        saveFrame(pathTarget=maskPath, image=mask, token=token, maskConversion=True)
    print(img[0].shape)
    return img, mask, token

def convertAll(img: ndarray, mask: ndarray, projectPath: str, token: str='converted', saveProgress: bool=True) -> ndarray:
    imgPath, maskPath = pathCreator(projectPath)
    if saveProgress:
        saveFrame(pathTarget=imgPath, image=img, token=token)
        saveFrame(pathTarget=maskPath, image=mask, token=token, maskConversion=True)
    return img, mask, token

def resizeAll(img: ndarray, projectPath: str = '', aspectRatio: tuple = None, mask: ndarray = [], token: str = 'resized', saveProgress: bool=True) -> ndarray:
    imagesResized = []
    maskResized = []
    aspectRatio = (512,512) if aspectRatio is None else aspectRatio
    if projectPath is not '': imgPath, maskPath = pathCreator(projectPath=projectPath)

    print('Image Array Size', img.shape)
    print('Mask Array Size', mask.shape)

    for singleFrame in img:
        resizedFrame = resizeSingleFrame(frame=singleFrame, aspectRatio=aspectRatio)
        imagesResized.append(resizedFrame)
    if saveProgress:
        saveFrame(pathTarget=imgPath, image=imagesResized, token=token)

    if mask is not None:
        for singleFrame in mask:
            resizedFrame = resizeSingleFrame(frame=singleFrame, aspectRatio=aspectRatio)
            maskResized.append(resizedFrame)
        if saveProgress:
            saveFrame(pathTarget=maskPath, image=maskResized, token=token, maskConversion=True) 
    return array(imagesResized), array(maskResized), token


#  =========================================
#  	        Offline Pre-Processing		
#  =========================================

def preProcStart(argument, projectPath, aspectRatio):
    img, mask = setupData(projectPath=projectPath, split=False)
    if argument[0] == 'align': alignAll(img=img, projectPath=projectPath)
    elif argument[0] == 'crop': cropAll(img=img, mask=mask, projectPath=projectPath)
    elif argument[0] == 'convert': convertAll(img=img, mask=mask, projectPath=projectPath)
    elif argument[0] == 'resize': resizeAll(img=img, mask=mask, projectPath=projectPath, aspectRatio=aspectRatio)
    elif argument[0] == 'training': preProcForDataStorage(frame=img, mask=mask, projectPath=projectPath, aspectRatio=aspectRatio)
    elif argument[0] == 'segment': preProcForSegment(imgArray=img, projectPath=projectPath, aspectRatio=aspectRatio)
    else: print('Unknown argument')
    print('\nFinished Pre-Processing')

def preProcFromCamera(frame: ndarray, aspectRatio: tuple = (512,512), channel: int = 1) -> ndarray:
    ''' Takes in an array of shape (2048,2448,3) from the camera live stream. 
    Returns (1,imWidth, imHeight, channel) array for segmentation.
    Applies for single images only. For multiple images, iterate over the function. '''
    frameCrop = cropImage(frame)
    frameResize = resizeSingleFrame(frame=frameCrop, aspectRatio=aspectRatio, channel=channel)
    return frameResize

def preProcForSegment(imgArray: ndarray, projectPath: str, aspectRatio = (512,512), channel: int = 1) -> None:
    ''' Takes in an array from the data storage and saves a subfolder 'segmentation' with pre-processed images.'''
    imgPath, maskPath = pathCreator(projectPath=projectPath)
    segmentImg = []
    for img in imgArray:
        upscaledFrame = resize(img, (2048,2448), interpolation=INTER_LINEAR)
        segmentFrame = preProcFromCamera(frame=upscaledFrame, aspectRatio=aspectRatio, channel=channel)
        segmentImg.append(segmentFrame)
    saveFrame(pathTarget=imgPath, image=segmentImg, token='segmentation')
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
    
def preProcForDataStorage(frame: ndarray, mask: ndarray, projectPath: str, aspectRatio = (512,512)) -> None:
    ''' Takes in high-resolution image frame and mask of shape (2048,2448,3) and (2048,2448) from 'img' subfolder.
    Saves images to subfolders of the project path after 1) cropping, 2) resizing.
    Resizing to target aspect ratio. Saves images and masks to pre-defined file formats in 'final' folder. '''
    print('\nCROPPING IMAGES')
    frameCrop, maskCrop, _ = cropAll(img=frame, mask=mask, projectPath=projectPath)
    print('\nRESIZING IMAGES')
    frameResize, maskResize, _ = resizeAll(img=frameCrop, mask=maskCrop, projectPath=projectPath, aspectRatio=aspectRatio)
    print('\nCONVERTING IMAGES')
    frameFinal, maskFinal, _ = convertAll(img=frameResize, mask=maskResize, projectPath=projectPath, token='final')

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