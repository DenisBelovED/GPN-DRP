from core.constants import IMAGE_SIZE
import tensorflow as tf
from numpy.random import random


class TrainAugmentation:
    def __call__(self, images_shape_list, images_list, labels_list):
        images_list = (1 / 255) * images_list
        if random() > 0.5:
            images_list = tf.image.flip_left_right(images_list)
        if random() > 0.6:
            images_list = tf.image.random_brightness(images_list, 0.3)
        if random() > 0.6:
            images_list = tf.image.random_contrast(images_list, 0.1, 0.5)
        if random() > 0.7:
            images_list = tf.image.random_hue(images_list, 0.25)
        if random() > 0.7:
            images_list = tf.image.random_saturation(images_list, 5, 10)
        return images_shape_list, images_list, labels_list


class InferenceAugmentation:
    def __call__(self, images_shape_list, images_list, labels_list):
        images_list = (1 / 255) * images_list
        return images_shape_list, images_list, labels_list
