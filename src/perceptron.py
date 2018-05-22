import tensorflow as tf

from src.NN import NN as NN

class Perceptron(NN):

    HIDDEN_NEURONS = (900, 400)

    def __init__(self, learning_rate, momentum, activation, filtered):
        NN.__init__(self, learning_rate, momentum, activation, filtered)

    def _create_nn(self, activation):
        initializer = tf.random_normal_initializer(0.0, 0.1)

        hidden = tf.layers.dense(self.input, self.HIDDEN_NEURONS[0],
                                 activation=activation,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer)
        hidden2 = tf.layers.dense(hidden, self.HIDDEN_NEURONS[1], activation=activation,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer)
        out = tf.layers.dense(hidden2, self.CLASSES, activation=tf.nn.softmax,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer)
        return out
