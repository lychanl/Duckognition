from src.NN import NN
import tensorflow as tf

class Conv(NN):
    TFRECORD_SUFFIX = '_50x50_proc'

    def __init__(self):
        NN.__init__(self, 0.01, 0.99, self.TFRECORD_SUFFIX, 50, 50, 3)

    def _create_nn(self):
        current = tf.layers.conv2d(self.X, 3, 16)
        current = tf.layers.max_pooling2d(current, 2, 2)
        current = tf.layers.conv2d(current, 3, 20)
        current = tf.layers.conv2d(current, 3, 20)
        current = tf.layers.max_pooling2d(current, 2, 2)
        current = tf.layers.conv2d(current, 5, 24)
        current = tf.layers.conv2d(current, 4, 28)
        current = tf.layers.dense(tf.reshape(current, [-1, 448]), 100)
        current = tf.layers.dense(current, 50)

        return current

