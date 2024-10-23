''' ## Header File

Defining static parameters. 
Also mimicking HTML interface while not connected.
'''
from os import getcwd, chdir
from os.path import join, dirname
from numpy import zeros, uint8

chdir(dirname(getcwd())) # move out of 'code' directory

# ======== Path Constants ========

CWD = getcwd() # path to current directory

REFERENCE_FOLDER = 'img'
IMAGE_LOAD_PATH = 'images' # source images
MASK_LOAD_PATH = 'masks' # source masks
SINGLE_DATA_PATH = join(CWD, 'example_images')
TRAIN_DATA_PATH = join(CWD, 'projects')
SAVE_MODEL_PATH = join(CWD, 'models')
SAVE_RES_PATH = join(CWD, 'results')

# ======== Data Containers ========

IMG_DATA = []
MASK_DATA = []
VAL_DATA = []

# ======== Threading Setup ========

IMG_SHAPE = (2048, 2448, 3)
IMG_ARRAY = zeros(IMG_SHAPE, dtype=uint8)
IMG_ARRAY.fill(255)
print('MAX IMG ARRAY: ', max(IMG_ARRAY))
 
# ======== Saving Settings ========

TIF_SUFFIX = '.tif'
PNG_SUFFIX = '.png'
MODEL_FORMAT = '.keras'
METRICS = ['loss', 'accuracy']
PROCESSES = ['milling', 'turning']