''' ## Cropping and align image routine

Accessible Functions:
- cropImage()
- alignImage()
'''

from numpy import ndarray, eye, float32
from cv2 import MOTION_TRANSLATION, TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT, INTER_LINEAR, WARP_INVERSE_MAP
from cv2 import findTransformECC, warpAffine

def cropImage(imageData: ndarray) -> ndarray:
    ''' Takes in image (and aspect ration) and crops the image. Returns cropped image'''
    for i in range(len(imageData)):  
        imgheight=imageData.shape[0] # DOES NOT MAKE SENSE YET; NEEDS PARAMETER INPUT
        imgwidth=imageData.shape[1]
        cropped_image = imageData[int(imgwidth/2)-1000:int(imgwidth/2)+1000, int(imgheight/2)-700:int(imgheight/2)+700] # Slicing to crop the image
    return cropped_image

def alignImage(imageData: ndarray) -> ndarray:
    ''' Takes in stack of images and aligns them according to the first image in the stack. Returns aligned stack.'''
    WARP_MODE = MOTION_TRANSLATION
    WARP_MATRIX = eye(2, 3, dtype=float32)
    NR_ITERATIONS = 10000
    TERMINATOR = 1e-10
    CRITERIA = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, NR_ITERATIONS, TERMINATOR) 
    sz = imageData[0].shape
    img_ref = imageData[0] # reference image (maybe save outside)
    for img in range(len(imageData)-1): 
        (_, WARP_MATRIX) = findTransformECC(img_ref,imageData[img+1],WARP_MATRIX, WARP_MODE, CRITERIA) 
        aligned_img = warpAffine(imageData[img], WARP_MATRIX, (sz[1],sz[0]), flags=INTER_LINEAR + WARP_INVERSE_MAP) 
    return aligned_img

