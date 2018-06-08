import sys
from src.svm import SVM
from src.conv import Conv
import tensorflow as tf

support_vectors = 2
gamma = 0.1

epochs = 250

use_softmax = True
use_cross_entropy = False
activation = tf.nn.sigmoid
layers_n = 9
descr_size = 32
name = r".\conv"
experiment = "base"

if len(sys.argv) > 1:
    epochs = float(sys.argv[1])

if len(sys.argv) > 2:
    experiment = sys.argv[2]
    if sys.argv[2] == "drop_layer":
        layers_n = 8
    elif sys.argv[2] == "add_layer":
        layers_n = 10
    elif sys.argv[2] == "softsign":
        activation = tf.nn.softsign
    elif sys.argv[2] == "relu":
        activation = tf.nn.relu
    elif sys.argv[2] == "incr_descr":
        descr_size = 48
    elif sys.argv[2] == "decr_descr":
        descr_size = 32
    elif sys.argv[2] == "sigmoid_cross_entropy":
        use_cross_entropy = True
        use_softmax = False
    elif sys.argv[2] == "softmax_cross_entropy":
        use_cross_entropy = True
        use_softmax = True
    else:
        raise Exception("Invalid experiment name!")

    name += "_" + sys.argv[2]

if len(sys.argv) > 3:
    gamma = float(sys.argv[3])

name += ".ckpt"
tf.set_random_seed(0)


tf.set_random_seed(0)
with tf.device('/cpu:0'), tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    print("Creating SVM, descriptor from experiment %s, number of support vectors = %d, gamma = %f, %d epochs"
          % (experiment, support_vectors, gamma, epochs))
    conv = Conv(layers_n, descr_size, activation, use_softmax, use_cross_entropy)

    tf.train.Saver().restore(sess, name)

    svm = SVM(conv, support_vectors, gamma)
    sess.run(tf.variables_initializer(svm.variables()))

    print("before:")
    svm.eval(sess)

    svm.train(sess, epochs=epochs)
