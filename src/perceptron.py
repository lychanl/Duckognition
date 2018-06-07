import tensorflow as tf

from src.NN import NN as NN

class Perceptron(NN):

    INPUT_WIDTH = 20
    INPUT_HEIGHT = 20
    INPUT_CHANNELS_FILTERED = 12
    # for filtered input should be like
    #   channels * width * height
    # instead of
    #   width * height * channels
    # but as we flatten input afterwards size is the only thing that matters
    INPUT_CHANNELS_UNFILTERED = 3
    HIDDEN_NEURONS = (900, 400)

    def __init__(self, learning_rate, momentum, activation, filtered):
        self._activation = activation
        tfrecord_suffix = '_proc' if filtered else '_unf_proc'
        self._channels = self.INPUT_CHANNELS_FILTERED if filtered else self.INPUT_CHANNELS_UNFILTERED

        NN.__init__(self, learning_rate, momentum, tfrecord_suffix, self.INPUT_WIDTH, self.INPUT_HEIGHT, self._channels)

    def _create_nn(self):
        initializer = tf.random_normal_initializer(0.0, 0.1)

        input = tf.reshape(self.input, [-1, self.INPUT_WIDTH * self.INPUT_HEIGHT * self._channels])

        hidden = tf.layers.dense(input, self.HIDDEN_NEURONS[0],
                                 activation=self._activation,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer)
        hidden2 = tf.layers.dense(hidden, self.HIDDEN_NEURONS[1], activation=self._activation,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer)
        out = tf.layers.dense(hidden2, self.CLASSES, activation=tf.nn.softmax,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer)
        return out
