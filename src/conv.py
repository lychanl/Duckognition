from src.NN import NN
import tensorflow as tf

class Conv(NN):
    TFRECORD_SUFFIX = '_50x50_proc'

    def __init__(self):
        NN.__init__(self, 0.01, 0.99, self.TFRECORD_SUFFIX, 50, 50, 3)

    def _create_nn(self):
        current = tf.layers.conv2d(self.input, 16, 3, activation=tf.nn.sigmoid)
        current = tf.layers.max_pooling2d(current, 2, 2)
        current = tf.layers.conv2d(current, 20, 3, activation=tf.nn.sigmoid)
        current = tf.layers.conv2d(current, 20, 3, activation=tf.nn.sigmoid)
        current = tf.layers.max_pooling2d(current, 2, 2)
        current = tf.layers.conv2d(current, 24, 5, activation=tf.nn.sigmoid)
        current = tf.layers.conv2d(current, 28, 3, activation=tf.nn.sigmoid)
        current = tf.layers.dense(tf.reshape(current, [-1, 448]), 100, activation=tf.nn.sigmoid)
        current = tf.layers.dense(current, 50, activation=tf.nn.softmax)

        return current

