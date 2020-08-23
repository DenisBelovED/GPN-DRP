import cv2
from os.path import normpath, join
from os import listdir
from core.constants import INFER_BATCH_SIZE, T_BATCH_SIZE, IMAGE_SIZE
import tensorflow as tf


class DataController:
    """
    Этот класс управляет подачей данных, содержит механизмы для аугментации "на лету".
    """
    def __init__(self, path_to_on_data, path_to_off_data, transform):
        self.on_data = [normpath(join(path_to_on_data, img_name)) for img_name in listdir(path_to_on_data)]
        self.off_data = [normpath(join(path_to_off_data, img_name)) for img_name in listdir(path_to_off_data)]
        self.ids = self.on_data + self.off_data
        self.dataset_indexes = tf.data.Dataset.range(len(self.ids))
        self.transform = transform
        self.autotune = tf.data.experimental.AUTOTUNE
        self.count_images = len(self.ids)

    def get_eval_data_generator(self, shuffle=False, inference=False):
        if shuffle:
            self.dataset_indexes = self.dataset_indexes.shuffle(len(self.ids))
        b_s = INFER_BATCH_SIZE if inference else T_BATCH_SIZE
        for indexes in self.dataset_indexes.batch(b_s, drop_remainder=True):
            image_sizes_list = []
            images_list = []
            labels_list = []
            for i in indexes.numpy():
                label = 1 if i < len(self.on_data) else 0
                image, img_h, img_w = self._read_image(self.ids[i])

                image_sizes_list.append((img_h, img_w))
                images_list.append(image)
                labels_list.append(label)
            image_sizes_list, images_list, labels_list = self.transform(
                image_sizes_list,
                tf.convert_to_tensor(images_list, dtype=tf.float32),
                labels_list
            )
            yield image_sizes_list, images_list, labels_list

    def get_reshuffle_data_generator(self):
        lambda_func = lambda x: tf.py_function(
            func=self._index_to_data_pair,
            inp=[x],
            Tout=[tf.int32, tf.float32, tf.int32]
        )

        """
        # uncomment for debugging data encoder
        for e in self.dataset_indexes.shuffle(len(self.ids)).batch(T_BATCH_SIZE, drop_remainder=True):
            yield lambda_func(e)
        """

        return \
            self.dataset_indexes \
                .shuffle(len(self.ids)) \
                .batch(T_BATCH_SIZE, drop_remainder=True) \
                .map(lambda_func, num_parallel_calls=self.autotune) \
                .prefetch(self.autotune)

    def _index_to_data_pair(self, indexes):
        image_sizes_list = []
        images_list = []
        labels_list = []

        for i in indexes.numpy():
            label = 1 if i < len(self.on_data) else 0
            image, img_h, img_w = self._read_image(self.ids[i])

            image_sizes_list.append((img_h, img_w))
            images_list.append(image)
            labels_list.append(label)

        with tf.device('CPU'):
            image_sizes_list, images_list, labels_list = self.transform(
                image_sizes_list,
                tf.convert_to_tensor(images_list, dtype=tf.float32),
                labels_list
            )

        return tf.convert_to_tensor(image_sizes_list, dtype=tf.int32), images_list, labels_list

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, h, w
