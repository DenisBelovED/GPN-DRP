import tensorflow as tf
from core.constants import T_BATCH_SIZE, INFER_BATCH_SIZE, NUM_CLASSES


class ProjectionBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, padding, stride, in_depth, out_depth):
        super(ProjectionBlock, self).__init__()
        self.proj_conv = tf.keras.layers.Conv2D(out_depth, kernel_size, stride, padding, activation='elu')
        self.conv_1 = tf.keras.layers.Conv2D(in_depth, (3, 3), 1, 'same', activation='elu')
        self.conv_2 = tf.keras.layers.Conv2D(in_depth, (3, 3), 1, 'same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        return self.proj_conv(tf.nn.elu(self.bn(self.conv_2(self.conv_1(x)) + x)))


class Block(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(Block, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(channels, (3, 3), 1, 'same', activation='elu')
        self.conv_2 = tf.keras.layers.Conv2D(channels, (3, 3), 1, 'same')

    def call(self, x):
        return tf.nn.elu(self.conv_2(self.conv_1(x)) + x)


class ResNet(tf.keras.Model):
    def __init__(self, inference_mode=False):
        self.batch_size = T_BATCH_SIZE
        if inference_mode:
            self.batch_size = INFER_BATCH_SIZE
        super(ResNet, self).__init__()
        # naming rule: self.BlockName_OutSize_Index
        self.p_block_194_1 = ProjectionBlock(7, 'valid', 1, 3, 3)
        self.p_block_188_1 = ProjectionBlock(7, 'valid', 1, 3, 3)
        self.p_block_182_1 = ProjectionBlock(7, 'valid', 1, 3, 3)
        self.p_block_180_1 = ProjectionBlock(3, 'valid', 1, 3, 8)
        self.block_180_1 = Block(8)
        #self.block_180_2 = Block(8)
        #self.block_180_3 = Block(8)
        self.p_block_90_1 = ProjectionBlock(3, 'same', 2, 8, 16)
        self.block_90_1 = Block(16)
        #self.block_90_2 = Block(16)
        #self.block_90_3 = Block(16)
        self.p_block_45_1 = ProjectionBlock(3, 'same', 2, 16, 24)
        self.block_45_1 = Block(24)
        #self.block_45_2 = Block(24)
        #self.block_45_3 = Block(24)
        self.p_block_23_1 = ProjectionBlock(3, 'same', 2, 24, 32)
        self.block_23_1 = Block(32)
        #self.block_23_2 = Block(32)
        #self.block_23_3 = Block(32)
        self.p_block_12_1 = ProjectionBlock(3, 'same', 2, 32, 40)
        self.block_12_1 = Block(40)
        #self.block_12_2 = Block(40)
        #self.block_12_3 = Block(40)
        self.p_block_6_1 = ProjectionBlock(3, 'same', 2, 40, 48)
        self.block_6_1 = Block(48)
        #self.block_6_2 = Block(48)
        #self.block_6_3 = Block(48)
        self.p_block_3_1 = ProjectionBlock(3, 'same', 2, 48, 56)
        self.block_3_1 = Block(56)
        #self.block_3_2 = Block(56)
        #self.block_3_3 = Block(56)
        self.out = tf.keras.layers.Dense(NUM_CLASSES)

    def call(self, inputs):
        x = self.p_block_194_1(inputs)
        x = self.p_block_188_1(x)
        x = self.p_block_182_1(x)

        x = self.p_block_180_1(x)
        x = self.block_180_1(x)
        #x = self.block_180_2(x)
        #x = self.block_180_3(x)

        x = self.p_block_90_1(x)
        x = self.block_90_1(x)
        #x = self.block_90_2(x)
        #x = self.block_90_3(x)

        x = self.p_block_45_1(x)
        x = self.block_45_1(x)
        #x = self.block_45_2(x)
        #x = self.block_45_3(x)

        x = self.p_block_23_1(x)
        x = self.block_23_1(x)
        #x = self.block_23_2(x)
        #x = self.block_23_3(x)

        x = self.p_block_12_1(x)
        x = self.block_12_1(x)
        #x = self.block_12_2(x)
        #x = self.block_12_3(x)

        x = self.p_block_6_1(x)
        x = self.block_6_1(x)
        #x = self.block_6_2(x)
        #x = self.block_6_3(x)

        x = self.p_block_3_1(x)
        x = self.block_3_1(x)
        #x = self.block_3_2(x)
        #x = self.block_3_3(x)

        return self.out(tf.reshape(x, (self.batch_size, 504)))
