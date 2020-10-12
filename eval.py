from core.constants import PATH_TO_MODEL_WEIGHT, IMAGE_SIZE, CHANNELS, DEVICE_TYPE, MEMORY_LIMIT, INFER_BATCH_SIZE, \
    WHITE_DICT, PATH_TO_TEST_DATA
import tensorflow as tf

# for debug:
# add here tf.config.experimental_run_functions_eagerly(True)
# or add tf.debugging.set_log_device_placement(True)
# add to environment variables CUDA_VISIBLE_DEVICES=""

# set level CUDA logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices(DEVICE_TYPE)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices(DEVICE_TYPE)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

print(f"Available GPUs:")
[print(f"\t{e}") for e in tf.config.list_physical_devices()]

from core.visual_manager import VisualManager
from core.data_controller import DataController
from core.augmentation_transforms import InferenceAugmentation
from core.resnet import ResNet
from numpy import concatenate
from time import time

VISUALIZATION = True


def main():
    if VISUALIZATION:
        window_manager = VisualManager('eval')

    dataset = DataController(
        PATH_TO_TEST_DATA,
        WHITE_DICT,
        transform=InferenceAugmentation(),
        is_validation_data=True,
        inference_mode=True
    )
    data_generator = dataset.get_eval_data_generator(shuffle=True)

    net = ResNet(inference_mode=True)
    net.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    net.load_weights(PATH_TO_MODEL_WEIGHT)
    net.summary()

    true_predicts = 0
    all_predicts = 0
    inf_time = 0

    for images, gt_labels in data_generator:
        gt_labels = gt_labels.numpy()
        t0 = time()
        pred_labels = tf.nn.softmax(net(images)).numpy().argmax(axis=1).astype('int32')
        inf_time += (time() - t0)

        all_predicts += INFER_BATCH_SIZE
        true_predicts += (gt_labels == pred_labels).sum()

        if VISUALIZATION:
            visual_batch = []
            for i in range(INFER_BATCH_SIZE):
                frame = (images[i] * 255).numpy().astype('uint8')
                window_manager.draw_annotations(frame, pred_labels[i], gt_labels[i], dataset.invert_class_dict)
                visual_batch.append(frame)
            window_manager.desktop_show(concatenate(visual_batch, axis=1), 0)

    print(
        f"Accuracy = {(true_predicts / all_predicts) * 100:.2f}% {true_predicts}/{all_predicts}\n"
        f"Median inference time: {(inf_time / all_predicts) * 1000:.1f} ms"
    )


if __name__ == '__main__':
    main()
