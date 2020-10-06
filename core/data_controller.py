import numpy as np
import cv2
from pickle import load
from os.path import normpath
from core.constants import ROOT_PREFIX, T_BATCH_SIZE, INFER_BATCH_SIZE, IMAGE_SIZE
import tensorflow as tf

'''
    Этот класс управляет подачей данных, содержит механизмы для аугментации "на лету".
'''


def buffer(get_generator):
    def iterable(obj):
        for images, labels in obj:
            yield tf.unstack(images), tf.unstack(labels)

    def wrapper(self):
        iterator = iterable(get_generator(self))
        img_buffer = []
        lbl_buffer = []
        while True:
            img_batch = img_buffer[:self.batch_size]
            lbl_batch = lbl_buffer[:self.batch_size]
            if len(img_batch) < self.batch_size:
                try:
                    images, labels = next(iterator)
                except StopIteration:
                    return
                img_buffer += images
                lbl_buffer += labels
            else:
                img_buffer = img_buffer[self.batch_size:]
                lbl_buffer = lbl_buffer[self.batch_size:]
                yield tf.stack(img_batch), tf.stack(lbl_batch)
    return wrapper


class DataController:
    def __init__(
            self,
            path_to_data,
            # путь до .pickle файла метаданных, описывающего датасет. Создаётся через скрипты в help_scripts.
            necessary_classes_dict,  # словарь с разрешёнными классами (WHITE_DICT в constants)
            transform=None,  # аугментация, которую нужно сделать
            is_validation_data=False,  # флажок для отключения количественных (но не классовых) ограничений в WHITE_DICT
            inference_mode=False
    ):
        with open(path_to_data, 'rb') as file:
            markup_data, summary, class_mapping = load(file)

        self.summary_object_count = sum(
            [summary[k] if (v == 0) or is_validation_data else v for k, v in necessary_classes_dict.items()]
        )

        if is_validation_data:
            necessary_classes_dict = dict.fromkeys(necessary_classes_dict.keys(), 0)

        for e in markup_data:
            if len(e['bboxes']) == 0:
                raise ValueError('bad dataset, annotation not found')
            else:
                for d in e['bboxes']:
                    if not d['class']:
                        raise ValueError('bad dataset, class not found in ' + e['filepath'])
                    if not ((d['x1'] >= 0) and (d['x2'] >= 0) and (d['y1'] >= 0) and (d['y2'] >= 0)):
                        raise ValueError('bad dataset, x1, x2, y1, y2 not found in ' + e['filepath'])
            if e['filepath'] == '':
                raise ValueError('bad dataset, full file path not found')

        # извлечение из датасета count объектов по каждому из классов, указанных в necessary_classes_dict
        necessary_markup_data = {}
        for e in markup_data:
            necessary_markup_data.update({e['filepath']: e.copy()})
            necessary_markup_data[e['filepath']]['bboxes'] = []

        new_summary_dict = dict.fromkeys(necessary_classes_dict.keys(), 0)

        for class_name in necessary_classes_dict.keys():
            if necessary_classes_dict[class_name] == 0:
                for e in markup_data:
                    necessary_bboxes = []
                    for bbox in e['bboxes']:
                        if bbox['class'] == class_name:
                            new_summary_dict[bbox['class']] += 1
                            necessary_bboxes.append(bbox)
                    if necessary_bboxes:
                        necessary_markup_data[e['filepath']]['bboxes'] += necessary_bboxes
            else:
                if necessary_classes_dict[class_name] < 0:
                    raise ValueError(f"количество объектов {class_name} должно быть >= 0")
                if necessary_classes_dict[class_name] > summary[class_name]:
                    raise ValueError(
                        f"указанное количество объектов {class_name} превышает возможное {summary[class_name]}")

                for e in markup_data:
                    necessary_bboxes = []
                    for bbox in e['bboxes']:
                        if (bbox['class'] == class_name) and (
                                new_summary_dict[bbox['class']] < necessary_classes_dict[class_name]):
                            new_summary_dict[bbox['class']] += 1
                            necessary_bboxes.append(bbox)
                    if necessary_bboxes:
                        necessary_markup_data[e['filepath']]['bboxes'] += necessary_bboxes

        self.ids = [v for k, v in necessary_markup_data.items() if v['bboxes']]
        self.class_dict = {}
        i = 0
        for label in necessary_classes_dict.keys():
            if label in class_mapping:
                self.class_dict.update({label: i})
                i += 1
        self.invert_class_dict = {v: k for k, v in self.class_dict.items()}
        self.classes_count = len(self.class_dict)
        self.summary = new_summary_dict
        self.transform = transform

        self.autotune = tf.data.experimental.AUTOTUNE
        self.dataset_indexes = tf.data.Dataset.range(len(self.ids))
        self.count_images_containing_necessary_classes = len(self.ids)
        self.batch_size = T_BATCH_SIZE
        if inference_mode:
            self.batch_size = INFER_BATCH_SIZE

        print(
            f"{new_summary_dict} "
            f"objects: {self.summary_object_count} images: {len(self.ids)}"
        )

    @buffer
    def get_eval_data_generator(self, shuffle=False):
        if shuffle:
            self.dataset_indexes = self.dataset_indexes.shuffle(len(self.ids))
        for indexes in self.dataset_indexes.batch(self.batch_size, drop_remainder=False):
            images_list = []
            labels_list = []
            for i in indexes.numpy():
                images, labels = self._get_annotation(self.ids[i])
                images_list.append(images)
                labels_list.append(labels)
            images_list, labels_list = self.transform(
                tf.convert_to_tensor(np.concatenate(images_list), dtype=tf.float32),
                np.concatenate(labels_list)
            )
            yield images_list, labels_list

    @buffer
    def get_reshuffle_data_generator(self):
        lambda_func = lambda x: tf.py_function(
            func=self._index_to_data_pair,
            inp=[x],
            Tout=[tf.float32, tf.int32]
        )

        """
        # uncomment for debugging data encoder
        for e in self.dataset_indexes.shuffle(len(self.ids)).batch(self.batch_size, drop_remainder=True):
            yield lambda_func(e)
        """

        return \
            self.dataset_indexes \
                .shuffle(len(self.ids)) \
                .batch(self.batch_size, drop_remainder=False) \
                .map(lambda_func, num_parallel_calls=self.autotune) \
                .prefetch(self.autotune)

    def _index_to_data_pair(self, indexes):
        images_list = []
        labels_list = []

        for i in indexes.numpy():
            images, labels = self._get_annotation(self.ids[i])
            images_list.append(images)
            labels_list.append(labels)

        with tf.device('CPU'):
            images_list, labels_list = self.transform(
                tf.convert_to_tensor(np.concatenate(images_list), dtype=tf.float32),
                np.concatenate(labels_list)
            )
        return images_list, labels_list

    def _get_annotation(self, image_id):
        image_path = normpath(ROOT_PREFIX + image_id['filepath'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = []
        images = []
        for object in image_id['bboxes']:
            images.append(
                cv2.resize(image[object['y1']:object['y2'], object['x1']:object['x2']], (IMAGE_SIZE, IMAGE_SIZE))
            )
            labels.append(self.class_dict[object['class']])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)
