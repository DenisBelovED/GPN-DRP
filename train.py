from core.constants import OFF_IMG_TEST_PATH, ON_IMG_TEST_PATH, OFF_IMG_TRAIN_PATH, ON_IMG_TRAIN_PATH, CHANNELS, \
    EPOCHS, IMAGE_SIZE, MODEL_WEIGHT_NAME, START_EPOCH, PATH_TO_MODEL_WEIGHT, CHECKPOINT_FOLDER, DEVICE_TYPE, \
    MEMORY_LIMIT, DEBUG_FOLDER, BATCH_SIZE

# for debug:
# tf.config.experimental_run_functions_eagerly(True) (if need)
# environment variables CUDA_VISIBLE_DEVICES=""

# from logging import disable, WARNING
# from os import environ
# disable(WARNING)
# environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# tf.debugging.set_log_device_placement(True)

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

from time import time
from core.augmentation_transforms import TrainAugmentation, InferenceAugmentation
from core.data_controller import DataController
from core.resnet import ResNet
from core.loss import CrossEntropyLoss
from os.path import join
import sys


def progress_bar(count, total, status):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f"\r{' ' * 100}\r{status}: [{bar}] {percents}%")
    sys.stdout.flush()


def main():
    # tf.random.set_seed(1337)
    tb_writer = tf.summary.create_file_writer(DEBUG_FOLDER)

    train_dataset = DataController(
        ON_IMG_TRAIN_PATH,
        OFF_IMG_TRAIN_PATH,
        transform=TrainAugmentation()
    )
    train_data_generator = train_dataset.get_reshuffle_data_generator()

    test_dataset = DataController(
        ON_IMG_TEST_PATH,
        OFF_IMG_TEST_PATH,
        transform=InferenceAugmentation()
    )
    test_data_generator = test_dataset.get_reshuffle_data_generator()

    net = ResNet()
    net.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    net.summary()

    if MODEL_WEIGHT_NAME:
        net.load_weights(PATH_TO_MODEL_WEIGHT)
        print("WEIGHTS LOADED")
    else:
        print("INIT NEW WEIGHTS")

    optimizer = tf.optimizers.Adam(1e-5)
    ce_loss = CrossEntropyLoss()

    history_train_step = 0
    history_test_step = 0
    for epoch in range(START_EPOCH, EPOCHS):
        epoch_train_loss = 0
        start_epoch_time = time()
        train_step = 1
        for image_sizes, images, gt_pairs in train_data_generator:
            with tf.GradientTape() as tape:
                logits = net(images)
                loss = ce_loss(gt_pairs, logits)
            gradients = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))

            epoch_train_loss += loss

            with tb_writer.as_default():
                # tf.summary.image('Images', images, history_train_step)
                tf.summary.scalar('Train step loss', loss, history_train_step)
                tb_writer.flush()

            progress_bar(
                train_step,
                train_dataset.count_images // BATCH_SIZE,
                'train'
            )
            train_step += 1
            history_train_step += 1

        print()

        test_step = 1
        epoch_test_loss = 0
        tf.profiler.experimental.start(DEBUG_FOLDER)
        for image_sizes, images, gt_pairs in test_data_generator:
            logits = net(images)
            loss = ce_loss(gt_pairs, logits)
            epoch_test_loss += loss

            with tb_writer.as_default():
                # tf.summary.image('Images', images, history_test_step)
                tf.summary.scalar('Test step loss', loss, history_test_step)
                tb_writer.flush()

            progress_bar(
                test_step,
                test_dataset.count_images // BATCH_SIZE,
                'test'
            )
            test_step += 1
            history_test_step += 1
        tf.profiler.experimental.stop()

        with tb_writer.as_default():
            tf.summary.scalar('Epoch train sum cls loss', epoch_train_loss, epoch)
            tf.summary.scalar('Epoch test sum cls loss', epoch_test_loss, epoch)
            tb_writer.flush()

        print(f"\nEpoch: {epoch}/{EPOCHS} {(time() - start_epoch_time) / 3600:.2f} h/e")

        net.save_weights(
            join(
                CHECKPOINT_FOLDER,
                f"ResNet-E{epoch}-C{int(epoch_train_loss)}.h5"
            ),
            save_format="h5"
        )

    tb_writer.close()


if __name__ == '__main__':
    main()
