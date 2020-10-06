import tensorflow as tf
from core.constants import NUM_CLASSES


class CrossEntropyLoss:
    def __call__(self, true_labels, cls_logits):
        one_hot_labels = tf.one_hot(true_labels, NUM_CLASSES, dtype=cls_logits.dtype)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(one_hot_labels), cls_logits)
        return tf.reduce_sum(cross_entropy)
