from core.constants import IMAGE_SIZE, CHANNELS, PATH_TO_MODEL_WEIGHT, DEVICE_TYPE, MEMORY_LIMIT
from data_preprocessing.frame_manager import FrameManager
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


class Inference:
    def __init__(self, stream_path):
        self.frame_generator = FrameManager.init_video_frame_generator(stream_path)

        self.net = ResNet(inference_mode=True)
        self.net.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        self.net.load_weights(PATH_TO_MODEL_WEIGHT)
        self.net.summary()
        self.aug = InferenceAugmentation()
        self.normalize = lambda frame: self.aug(None, frame, None)[1]

    def __call__(self):
        while True:
            try:
                frame = next(self.frame_generator)
                frame = tf.image.resize(frame[None, ...], (IMAGE_SIZE, IMAGE_SIZE))
                scores = tf.nn.softmax(self.net(self.normalize(frame))).numpy()[0]
                label = int(scores[0] < scores[1])
                res = (frame.numpy()[0], f"Indicator: {'ON' if label else 'OFF'} {scores[label] * 100:.2f}%")
                yield res
            except StopIteration:
                break
