import perceptron as p
import tensorflow as tf
import sys

# usage: python main.py lr mom act epoch
# 0.0 < lr < 1.0
# 0.0 < mom < 1.0
# act - string from {relu, relu6, crelu, elu, selu, softplus, softsign, dropout, bias_add, sigmoid, tanh, leaky_relu}
# epoch > 0 integer

learning_rate = 0.002
momentum = 0.99
activation = tf.nn.sigmoid
epochs = 10

if len(sys.argv) > 1 and 0.0 < float(sys.argv[1]) < 1.0:
    learning_rate = float(sys.argv[1])

if len(sys.argv) > 2 and 0.0 < float(sys.argv[2]) < 1.0:
    momentum = float(sys.argv[2])

if len(sys.argv) > 3:
    if sys.argv[3] == "relu":
        activation = tf.nn.relu
    elif sys.argv[3] == "relu6":
        activation = tf.nn.relu6
    elif sys.argv[3] == "crelu":
        activation = tf.nn.crelu
    elif sys.argv[3] == "elu":
        activation = tf.nn.elu
    elif sys.argv[3] == "selu":
        activation = tf.nn.selu
    elif sys.argv[3] == "leaky_relu":
        activation = tf.nn.leaky_relu
    elif sys.argv[3] == "softplus":
        activation = tf.nn.softplus
    elif sys.argv[3] == "softsign":
        activation = tf.nn.softsign
    elif sys.argv[3] == "dropout":
        activation = tf.nn.dropout
    elif sys.argv[3] == "bias_add":
        activation = tf.nn.bias_add
    elif sys.argv[3] == "sigmoid":
        activation = tf.nn.sigmoid
    elif sys.argv[3] == "tanh":
        activation = tf.nn.tanh

if len(sys.argv) > 4 and int(sys.argv[4]) > 0:
    epochs = int(sys.argv[4])


with tf.Session() as sess:
    print("Creating perceptron, learning rate=%f, momentum=%f, activation=%s. %d epochs." % (learning_rate, momentum,
          str(activation), epochs))
    perceptron = p.Perceptron(learning_rate=learning_rate, momentum=momentum, activation=activation)

    sess.run(tf.global_variables_initializer())

    print("before:")
    perceptron.eval(sess)

    perceptron.train(sess, epochs=epochs)
