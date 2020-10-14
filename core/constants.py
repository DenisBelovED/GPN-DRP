from os import getcwd, mkdir
from os.path import join, exists, normpath
from shutil import rmtree

# используемый девайс
DEVICE_TYPE = 'GPU'

# предел используемой памяти девайса в мегабайтах
MEMORY_LIMIT = 1000

# Размер пакета, размер входной картинки
INFER_BATCH_SIZE = 10
T_BATCH_SIZE = 65
IMAGE_SIZE = 200
CHANNELS = 3
EPOCHS = 1000

# устанавливаем префикс до папки проекта
ROOT_PREFIX = getcwd()
ROOT_PREFIX = ROOT_PREFIX[:ROOT_PREFIX.index(r'people_classifier')]

# Лог для отладки в tensorboard
DEBUG_FOLDER = normpath(join(ROOT_PREFIX, r'people_classifier/src/debug/'))
if not exists(DEBUG_FOLDER):
    mkdir(DEBUG_FOLDER)
else:
    rmtree(DEBUG_FOLDER)
    mkdir(DEBUG_FOLDER)

# путь до папки, где сохраняются/читаются веса
CHECKPOINT_FOLDER_PATH = r'people_classifier/src/models/'
CHECKPOINT_FOLDER = normpath(join(ROOT_PREFIX, CHECKPOINT_FOLDER_PATH))
if not exists(CHECKPOINT_FOLDER):
    mkdir(CHECKPOINT_FOLDER)

# файлы описания данных
PATH_TO_TRAIN_DATA = normpath(join(ROOT_PREFIX, r'TF_2.2_SSD/data/prepared_data/train_full_extended_dataset.pickle'))
PATH_TO_TEST_DATA = normpath(join(ROOT_PREFIX, r'TF_2.2_SSD/data/prepared_data/test_azs_dataset.pickle'))
PATH_TO_DEBUG_DATA = normpath(join(ROOT_PREFIX, r'TF_2.2_SSD/data/prepared_data/debug_dataset.pickle'))

# имена весов и логика продолжения/инициализации обучения
MODEL_WEIGHT_NAME = r''
PATH_TO_MODEL_WEIGHT = normpath(join(ROOT_PREFIX, CHECKPOINT_FOLDER_PATH, MODEL_WEIGHT_NAME))
START_EPOCH = 1
if MODEL_WEIGHT_NAME:
    START_EPOCH = int(MODEL_WEIGHT_NAME.split(sep='-')[1][1:]) + 1
else:
    PATH_TO_MODEL_WEIGHT = None

"""
WHITE_DICT
    Этот словарь отвечает за порядок перечисления и количество объектов класса.
    Если указать при треннировке классы в данном порядке, то для правильного декодирования после инференса
    порядок изменять нельзя.
    Количество классов при инференсе должно совпадать с количеством при обучении, а разные имена допустимы.
    Имена классов должны быть подмножеством множества summary, которое получаем из файла описания датасета.
    Класс DataController подгрузит файл описания датасета, и выведет summary в консоль.  
    Числами задаётся первое N объектов класса, которые встретятся в датасете. 
    Если число = 0 то используются все объекты класса.
    Если передать в DataController данный словарь с параметром is_validation_data=True,
    то число объектов будет проигнорированно, и задействуются все объекты.
"""
WHITE_DICT = {
    # 'auto': 60000,
    # 'fleet': 0,
    # 'CRT': 0,
    'person': 16000,
    'operator': 0,
    'cashier': 0
}
NUM_CLASSES = len(WHITE_DICT)
