from os import getcwd, mkdir
from os.path import join, exists, normpath
from shutil import rmtree

# используемый девайс
DEVICE_TYPE = 'GPU'

# предел используемой памяти девайса в мегабайтах
MEMORY_LIMIT = 512

# Размер пакета, размер входной картинки
INFER_BATCH_SIZE = 4
T_BATCH_SIZE = 3
IMAGE_SIZE = 360
CHANNELS = 3
EPOCHS = 1000

# устанавливаем префикс до папки проекта
ROOT_PREFIX = getcwd()
ROOT_PREFIX = ROOT_PREFIX[:ROOT_PREFIX.index(r'rosseti')]

# Лог для отладки в tensorboard
DEBUG_FOLDER = normpath(join(ROOT_PREFIX, r'rosseti/src/debug/'))
if not exists(DEBUG_FOLDER):
    mkdir(DEBUG_FOLDER)
else:
    rmtree(DEBUG_FOLDER)
    mkdir(DEBUG_FOLDER)

# путь до папки, где сохраняются/читаются веса
CHECKPOINT_FOLDER_PATH = r'rosseti/src/models/'
CHECKPOINT_FOLDER = normpath(join(ROOT_PREFIX, CHECKPOINT_FOLDER_PATH))
if not exists(CHECKPOINT_FOLDER):
    mkdir(CHECKPOINT_FOLDER)

# Пути до данных
OFF_VIDEO_PATH = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/off_video'))
ON_VIDEO_PATH = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/on_video'))
OFF_IMG_TRAIN_PATH = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/off'))
ON_IMG_TRAIN_PATH = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/on'))
OFF_IMG_TEST_PATH = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/off_test'))
ON_IMG_TEST_PATH = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/on_test'))

# имена весов и логика продолжения/инициализации обучения
MODEL_WEIGHT_NAME = r'ResNet-E29-C126.h5'
PATH_TO_MODEL_WEIGHT = join(ROOT_PREFIX, CHECKPOINT_FOLDER_PATH, MODEL_WEIGHT_NAME)
START_EPOCH = 1
if MODEL_WEIGHT_NAME:
    START_EPOCH = int(MODEL_WEIGHT_NAME.split(sep='-')[1][1:]) + 1
else:
    PATH_TO_MODEL_WEIGHT = None
