'''## Offline Data Preprocessing

Processing raw image data by creating specific folders for the image manipulation task.
Takes in 'Project Name' the images are filed for.

Accessible Functions:
- offlinePreProc()
'''

from numpy import ndarray, unique, linspace
from generalUtensils import imageReader, setupData, saveImage, pathCreator
import augmentation as aug
import crop_align as crop
from cv2 import resize, INTER_LINEAR

def initPreProc(projectPath) -> ndarray:
    ''' Preprocessing works with token passing to ensure the right suffix for the process. 

    Image processing -> align, crop, convert to tif, augmentation.'''
    global imgPath, maskPath # expose paths to other functionss
    imgPath, maskPath = pathCreator(projectPath=projectPath)
    thisToken = 'start' # initial token
    img, mask = setupData(projectPath=projectPath, split=False)
    return img, mask, thisToken

def alignAll(img: ndarray, mask: ndarray, token=None) -> ndarray:
    ''' Aligning stack of images and masks. '''
    thisToken= 'aligned'
    img = crop.alignImage(imageData=img)
    mask = crop.alignImage(imageData=mask)

    saveImage(pathTarget=imgPath, image=img, token=thisToken) # save images
    saveImage(pathTarget=maskPath, image=mask, token=thisToken) # save masks
    return img, mask, thisToken

def augmentAll(img: ndarray, mask: ndarray, token=None) -> ndarray:
    ''' Augment stack of images and masks. '''
    thisToken= 'augmented'
    img, mask = aug.imageAugmentation(img, mask)

    saveImage(pathTarget=imgPath, image=img, token=thisToken) # save images
    saveImage(pathTarget=maskPath, image=mask, token=thisToken) # save masks
    return img, mask, thisToken

def cropAll(img: ndarray, mask: ndarray, token=None) -> ndarray:
    ''' Crop stack of images and masks. '''
    thisToken= 'cropped'
    img = imageReader(imgPath)  # read in images
    mask = imageReader(maskPath)# read in masks
    img = crop.cropImage(img)
    mask = crop.cropImage(mask)

    saveImage(pathTarget=imgPath, image=img, token=thisToken) # save images
    saveImage(pathTarget=maskPath, image=mask, token=thisToken) # save masks
    return img, mask, thisToken

def convertAll(img: ndarray, mask: ndarray, token=None) -> ndarray:
    ''' Convert file type of stack of images / masks 
    
    HAS ERROR! '''
    thisToken= 'converted'

    # img = Image.fromarray(img)
    # mask = Image.fromarray(mask)
    # # setting up folders

    # # convert images
    # setupFolder(path=imgPath, token=thisToken)
    # output_folder = os.path.join()
    # for imgName in os.listdir(imgPath):
    #     output_path = os.path.join(output_folder, os.path.splitext(imgName)[0] + hd.IMG_TARGET_SUFFIX)
    #     img.save(output_path, hd.IMG_TARGET)
    
    # img = imageReader(output_folder)

    # # convert masks
    # setupFolder(path=maskPath, token=thisToken)
    # for maskEntry in mask:
    #     print(i)
    #     i-=1

    return img, mask, thisToken


def offlinePreProc(projectPath: str) -> ndarray:
    ''' Start offline pre-processing. Takes in path to project file in which the images lie that are processed. '''
    img, mask, token = initPreProc(projectPath=projectPath) # initializing image pre-processing by loading the data
    # img, mask, token= alignAll(img, mask, token) # 1st step: alignment
    img, mask, token= augmentAll(img, mask, token) # 2nd step: augmentation
    img, mask, token= convertAll(img, mask, token) # 3rd step convert
    img, mask, token= cropAll(img, mask, token) # 4th step: crop
    
    saveImage(pathTarget=imgPath, image=img, token='final') # save final images
    saveImage(pathTarget=maskPath, image=mask, token='final') # save final masks
    return img, mask

def resizeImage(parameters: dict, data: ndarray) -> ndarray:
    ''' Resize images (image data) according to parameters. Returns stack of resized images.'''
    height = int(parameters['ImageHeight_int'])
    width = int(parameters['ImageWidth_int'])
    aspect = (height, width)
    resizeImages = []
    for img in data:
        resized = resize(img, aspect, interpolation=INTER_LINEAR)
        resizeImages.append(resized)
    return resizeImages

def maskConversion(masks, numClasses):
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
            masks[(threshold[c]<= masks)&threshold[c+1]>masks] = colorClasses[c]
    return masks