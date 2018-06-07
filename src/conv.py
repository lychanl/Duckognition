from src.NN import NN
import tensorflow as tf

class Conv(NN):
    TFRECORD_SUFFIX = '_50x50_proc'

    def __init__(self, layers_n=9, layer_7_filters_n=40, activation=tf.nn.sigmoid, use_softmax=True, cross_entropy=False, train_batch_size=5000):

        assert(layers_n in (8, 9, 10))

        loss = tf.losses.softmax_cross_entropy if use_softmax else tf.losses.sigmoid_cross_entropy

        self.activation = activation
        self.last_activation = None if cross_entropy else tf.nn.softmax if use_softmax else None

        self.layer_7_filters_n = layer_7_filters_n
        self.drop_conv = layers_n == 8
        self.add_conv = layers_n == 10

        NN.__init__(self, 0.01, 0.99, self.TFRECORD_SUFFIX, 50, 50, 3, loss if cross_entropy else None, train_batch_size)

    def _create_nn(self):
        current = tf.layers.conv2d(self.input, 20, 3, activation=self.activation)
        current = tf.layers.max_pooling2d(current, 2, 2)
        current = tf.layers.conv2d(current, 24, 3, activation=self.activation)
        current = tf.layers.conv2d(current, 24, 3, activation=self.activation)
        current = tf.layers.max_pooling2d(current, 2, 2)
        self.descriptor = current = tf.layers.conv2d(current, 32, 5, activation=self.activation)
        if not self.drop_conv:
            current = tf.layers.conv2d(current, self.layer_7_filters_n, 3, activation=self.activation)
        if self.add_conv:
            current = tf.layers.conv2d(current, 48, 2, activation=self.activation)

        self.descriptor = current

        last_conv_output_shape = current.get_shape().as_list()
        self.descr_size = last_conv_output_shape[1] * last_conv_output_shape[2] * last_conv_output_shape[3]

        current = tf.layers.dense(tf.reshape(current, [-1, self.descr_size]), 100, activation=self.activation)
        current = tf.layers.dense(current, 50, activation=self.last_activation)

        return current

