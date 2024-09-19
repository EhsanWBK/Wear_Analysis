''' ## Data Augmentation Routines

Inspired by: Sreenivas Bhattiprolu
https://youtu.be/k4TqxHteJ7s
https://youtu.be/mwN2GGA4mqo

Accessible Functions: 
- imageAugmentation()
'''

from numpy import fliplr, flipud, roll, ndarray
from random import seed as sd, randint
from scipy.ndimage import rotate

from generalUtensils import imageReader

# augmentation operations

def rotation(image, seed):
    sd(seed)
    angle= 180
    r_img = rotate(image, angle, mode='reflect', reshape=False, order=0)
    return r_img

def h_flip(image, seed):
    hflipped_img= fliplr(image)
    return  hflipped_img

def v_flip(image, seed):
    vflipped_img= flipud(image)
    return vflipped_img

def v_transl(image, seed):
    sd(seed)
    n_pixels = randint(-64,64)
    vtranslated_img = roll(image, n_pixels, axis=0)
    return vtranslated_img

def h_transl(image, seed):
    sd(seed)
    n_pixels = randint(-64,64)
    htranslated_img = roll(image, n_pixels, axis=1)
    return htranslated_img

transformations = { 'rotate': rotation,
                    'horizontal flip': h_flip, 
                    'vertical flip': v_flip
                    }

def imageAugmentation(images: ndarray, masks: ndarray) -> ndarray:
    ''' Takes in two numpy array of multiple images and masks and returns these after applying augmentation.'''
    imageArray = []
    maskArray = []
    for i in range(len(images)): 
        image = images[i]
        mask = masks[i]        
        for n in range(len(transformations)):
            key = list(transformations)[n] #randomly choosing method to call
            seed = randint(1,100)  #Generate seed to supply transformation functions. 
            transformed_image = transformations[key](image,seed)
            transformed_mask = transformations[key](mask,seed)
            imageArray.append(transformed_image)
            maskArray.append(transformed_mask)
    return imageArray, maskArray