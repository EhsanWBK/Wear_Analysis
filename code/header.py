''' ## Header File

Defining static parameters. 
Also mimicking HTML interface while not connected.
'''
from os import getcwd
from os.path import join

# Path Constants
CWD = getcwd() # path to cur rent directory
REFERENCE_FOLDER = 'img'
IMAGE_LOAD_PATH = 'images' # source images
MASK_LOAD_PATH = 'masks' # source masks
TRAIN_DATA_PATH = join(CWD, 'data_training')
SAVE_MODEL_PATH = join(CWD, 'models')

# Containers
IMG_DATA = []
MASK_DATA = []
VAL_DATA = []
 
# Static Settings
IMG_TARGET = 'TIFF' # only needed for PIL saving
MASK_TARGET = 'PNG' # only needed for PIL saving
IMG_TARGET_SUFFIX = '.tif'
MASK_TARGET_SUFFIX = '.png'
MODEL_FORMAT = '.keras'
METRICS = ['loss', 'accuracy']
PROCESSES = ['milling', 'turning']

# Gloabl Variables (initialization; later overwritten)
currentModel = None 
# currentImage = None 
currentGroundTruth = None 
modelPath = None
projectName = None # path to project from; e.g.: ('milling', 'milling_1')
cameraConnection = False # video stream variable



# --------------------------------------------------------------------------------------------------
# ----------------------------- HTML - ARTIFICIAL INTERFACE ----------------------------------------
# --------------------------------------------------------------------------------------------------

# parameters for model training: artificial interface of HTML platform
parDict = {
    'batch_size': 10,
    'verbose': 1,
    'epochs': 10,
    'shuffle': False
}

# temporarily static
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_SHAPE = (512, 512, 1)

# changeable parameters
project_name = join('milling','milling_1')
data_path = join(TRAIN_DATA_PATH, project_name) # selectable over HTML surface; needs current directory
model_path = join(SAVE_MODEL_PATH, project_name, 'test_ehsan_augmentation.hdf5')
model_save_path = join(SAVE_MODEL_PATH, project_name)

TEST_SIZE = 0.1
SPLIT_RANDOM_STATE = 0

# preprocessing steps
DEFAULT_PP = ['align', 'crop'] # default online pre-processing steps

print(data_path)