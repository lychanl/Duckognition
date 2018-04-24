import perceptron as p
import tensorflow as tf


with tf.Session() as sess:
    percetron = p.Perceptron(learning_rate=0.002, momentum=0.99, activation=tf.sigmoid)

    sess.run(tf.global_variables_initializer())

    print("before:")
    percetron.eval(sess)

    percetron.train(sess, epochs=100)
