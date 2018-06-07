import sys
from src.lenet import LeNet
import tensorflow as tf

learning_rate = 0.001
momentum = 0.99
epochs = 250
filtered = True

if len(sys.argv) > 1 and 0.0 < float(sys.argv[1]) < 1.0:
    learning_rate = float(sys.argv[1])

if len(sys.argv) > 2 and 0.0 < float(sys.argv[2]) < 1.0:
    momentum = float(sys.argv[2])

if len(sys.argv) > 3:
    epochs = float(sys.argv[3])

tf.set_random_seed(0)
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    print("Creating LeNet, learning rate=%f, momentum=%f, %d epochs" % (learning_rate, momentum, epochs))
    lenet = LeNet(learning_rate=learning_rate, momentum=momentum)

    sess.run(tf.global_variables_initializer())

    print("before:")
    lenet.eval(sess)

    lenet.train(sess, epochs=epochs)
