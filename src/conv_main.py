import sys
from src.conv import Conv
import tensorflow as tf

epochs = 250
filtered = True

if len(sys.argv) > 1 and 0.0 < float(sys.argv[1]) < 1.0:
    learning_rate = float(sys.argv[1])

if len(sys.argv) > 2 and 0.0 < float(sys.argv[2]) < 1.0:
    momentum = float(sys.argv[2])

if len(sys.argv) > 3:
    epochs = float(sys.argv[3])

tf.set_random_seed(0)
with tf.device('/cpu:0'), tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    print("Creating Conv, %d epochs" % epochs)
    conv = Conv()

    sess.run(tf.global_variables_initializer())

    print("before:")
    conv.eval(sess)

    conv.train(sess, epochs=epochs)
