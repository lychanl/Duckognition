import tensorflow as tf
import math
from src.NN import NN

class LeNet(NN):
    TFRECORD_SUFFIX = '_50x50_proc'

    INPUT_WIDTH = 50
    INPUT_HEIGHT = 50

    FILTERS1_N = 6
    FILTERS1_SIZE = 4

    POOL1_SIZE = 2
    POOL1_STRIDES = 2

    FILTERS2_N = 16
    FILTERS2_SIZE = 5

    POOL2_SIZE = 2
    POOL2_STRIDES = 2

    POOL2_OUTPUT_WIDTH = (((INPUT_WIDTH - FILTERS1_SIZE + 1 - POOL1_SIZE) // POOL1_STRIDES + 1)
                                        - FILTERS2_SIZE + 1 - POOL2_SIZE) // POOL2_STRIDES + 1
    POOL2_OUTPUT_HEIGHT = (((INPUT_HEIGHT - FILTERS1_SIZE + 1 - POOL1_SIZE) // POOL1_STRIDES + 1)
                                          - FILTERS2_SIZE + 1 - POOL2_SIZE) // POOL2_STRIDES + 1

    HIDDEN_INPUT_SIZE = POOL2_OUTPUT_HEIGHT * POOL2_OUTPUT_WIDTH * FILTERS2_N
    HIDDEN_SIZE = 140

    def __init__(self, learning_rate, momentum):
        NN.__init__(self, learning_rate, momentum, self.TFRECORD_SUFFIX, self.INPUT_WIDTH, self.INPUT_HEIGHT, 3)

    def _create_nn(self):
        conv1 = tf.layers.conv2d(self.input, self.FILTERS1_N, self.FILTERS1_SIZE, activation=tf.nn.sigmoid)
        pool1 = tf.layers.max_pooling2d(conv1, self.POOL1_SIZE, self.POOL1_STRIDES)
        conv2 = tf.layers.conv2d(pool1, self.FILTERS2_N, self.FILTERS2_SIZE, activation=tf.nn.sigmoid)
        pool2 = tf.layers.max_pooling2d(conv2, self.POOL2_SIZE, self.POOL2_STRIDES)
        hidden = tf.layers.dense(tf.reshape(pool2, [-1, self.HIDDEN_INPUT_SIZE]),
                                 self.HIDDEN_SIZE, activation=tf.nn.sigmoid)
        return tf.layers.dense(hidden, self.CLASSES, activation=tf.nn.softmax)

