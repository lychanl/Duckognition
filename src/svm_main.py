import sys
from src.svm import SVM
from src.conv import Conv
import tensorflow as tf

name = 'conv.ckpt'

support_vectors = 1
epochs = 250

tf.set_random_seed(0)
with tf.device('/cpu:0'), tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    print("Creating SVM, number of support vectors = %d, %d epochs" % (support_vectors, epochs))
    conv = Conv(train_batch_size=1)

    sess.run(tf.global_variables_initializer())
    tf.train.Saver().restore(sess, name)

    svm = SVM(conv, support_vectors)
    sess.run(tf.variables_initializer(svm.varaibles()))


    print("before:")
    svm.eval(sess)

    svm.train(sess, epochs=epochs)
