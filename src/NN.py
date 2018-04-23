import tensorflow as tf

learning_rate = 0.02
momentum = 0.99
training_epochs = 100
batch_size = 42
display_step = 1

neurons = [625, 100, 75, 50]
classes = 50

X = tf.placeholder("float", [None, neurons[0]])
Y = tf.placeholder("float", [None, classes])

# .batch(batch_size).repeat(training_epochs)
train_data = tf.data.TFRecordDataset(['data_train.tfrecord']).map(lambda X, Y: (X, Y))
val_data = tf.data.TFRecordDataset(['data_validate.tfrecord']).map(lambda X, Y: (X, Y))

def mlp(activation):
    l1 = tf.layers.dense(X, neurons[1], activation=activation)
    l2 = tf.layers.dense(l1, neurons[2], activation=activation)
    out = tf.layers.dense(l2, neurons[3], activation=tf.nn.softmax)
    return out


nn = mlp(tf.nn.sigmoid)

loss_function = tf.reduce_sum(tf.squared_difference(nn, Y))
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
train_op = optimizer.minimize(loss_function)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(training_epochs):
        train_it = train_data.make_one_shot_iterator()
        val_it = val_data.make_one_shot_iterator()

        for data in train_it:
            pass  #   sess.run(train_op, {X: data.})



