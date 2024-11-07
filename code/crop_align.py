''' ## Cropping and align image routine

Accessible Functions:
- cropImage()
- alignImage()
'''

from numpy import ndarray, eye, float32
from cv2 import MOTION_TRANSLATION, TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT, INTER_LINEAR, WARP_INVERSE_MAP
from cv2 import findTransformECC, warpAffine

def cropImage(imageData: ndarray, deltaX: int = 700, deltaY: int = 1000) -> ndarray:
    ''' Takes in image (and aspect ration) and crops the image. Returns cropped image'''
    for i in range(len(imageData)): # pass one or multiple images  
        imgheight=imageData.shape[0]
        imgwidth=imageData.shape[1]

        # Ensure the cropping coordinates are within the image dimensions
        start_x = max(0, int(imgwidth/2) - deltaX)
        end_x = min(imgwidth, int(imgwidth/2) + deltaX)
        start_y = max(0, int(imgheight/2) - deltaY)
        end_y = min(imgheight, int(imgheight/2) + deltaY)

        # Slicing to crop the image
        cropped_image = imageData[start_y:end_y, start_x:end_x] 
    return cropped_image

def alignImage(imageData: ndarray) -> ndarray:
    ''' Takes in stack of images and aligns them according to the first image in the stack. Returns aligned stack.'''
    print('\nStarting Alignment:')
    WARP_MODE = MOTION_TRANSLATION
    WARP_MATRIX = eye(2, 3, dtype=float32)
    NR_ITERATIONS = 10000
    TERMINATOR = 1e-10
    CRITERIA = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, NR_ITERATIONS, TERMINATOR) 
    sz = imageData[0].shape
    imageData = imageData.astype('float32')
    print('Data Type: ',imageData.dtype)
    img_ref = imageData[0] # reference image (maybe save outside)
    print(img_ref.shape)
    try:
        for img in range(len(imageData)-1): 
            (_, WARP_MATRIX) = findTransformECC(img_ref,imageData[img+1],WARP_MATRIX, WARP_MODE, CRITERIA) 
            aligned_img = warpAffine(imageData[img], WARP_MATRIX, (sz[1],sz[0]), flags=INTER_LINEAR + WARP_INVERSE_MAP) 
    except:
        print('Warning: find transform failed.')
        return None, False
    print('Finished Alignment')
    return aligned_img, True

