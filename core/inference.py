from core.constants import IMAGE_SIZE, CHANNELS, PATH_TO_MODEL_WEIGHT, DEVICE_TYPE, MEMORY_LIMIT, INFER_BATCH_SIZE, \
    ON_IMG_TEST_PATH, OFF_IMG_TEST_PATH
from core.data_controller import DataController
from cv2 import putText, FONT_HERSHEY_SIMPLEX, LINE_8
import tensorflow as tf

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

from core.resnet import ResNet
from core.augmentation_transforms import InferenceAugmentation
from numpy import concatenate


class Inference:
    def __init__(self):
        self.net = ResNet(inference_mode=True)
        self.net.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        self.net.load_weights(PATH_TO_MODEL_WEIGHT)
        self.net.summary()

        self.dataset = DataController(
            ON_IMG_TEST_PATH,
            OFF_IMG_TEST_PATH,
            transform=InferenceAugmentation()
        )
        self.data_generator = self.dataset.get_eval_data_generator(shuffle=True, inference=True)

    def __call__(self):
        for image_sizes, images, gt_labels in self.data_generator:
            scores = tf.nn.softmax(self.net(images)).numpy()
            frame = (images * 255).numpy()[..., ::-1].astype('uint8')
            for i in range(INFER_BATCH_SIZE):
                label = int(scores[i][0] < scores[i][1])
                s, color = ('ON', (0, 255, 0)) if label else ('OFF', (0, 0, 255))
                putText(
                    frame[i], f"Sensor {i + 1}:     {scores[i][label] * 100:.2f}%",
                    (5, IMAGE_SIZE - 20), FONT_HERSHEY_SIMPLEX, 1, (0, 225, 255), 2, LINE_8
                )
                putText(frame[i], s, (160, IMAGE_SIZE - 20), FONT_HERSHEY_SIMPLEX, 1, color, 2, LINE_8)
            yield concatenate(frame, 1)
