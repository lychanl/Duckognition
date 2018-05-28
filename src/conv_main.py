import sys
from src.conv import Conv
import tensorflow as tf

epochs = 5000
filtered = True

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

name += ".ckpt"
tf.set_random_seed(0)


with tf.device('/cpu:0'), tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    print("Creating Conv, experiment: %s, %d epochs" % (experiment, epochs))
    conv = Conv()

    sess.run(tf.global_variables_initializer())

    print("before:")
    conv.eval(sess)

    conv.train(sess, epochs=epochs)

    saver = tf.train.Saver()
    print("Saved as: " + saver.save(sess, name))


