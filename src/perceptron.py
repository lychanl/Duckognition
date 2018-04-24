import tensorflow as tf


class Perceptron:

    INPUTS = 25 * 25 * 3
    CLASSES = 50
    NEURONS = (INPUTS, 100, 75, CLASSES)

    _feature = {
        'image/data': tf.VarLenFeature(tf.string),
        'image/class': tf.FixedLenFeature([], tf.int64),
    }

    def __init__(self, learning_rate, momentum, activation):
        self._X = tf.placeholder(tf.float32, [None, self.NEURONS[0]])
        self._Y = tf.placeholder(tf.float32, [None, self.CLASSES])
        self._train_data = tf.data.TFRecordDataset(['data_train.tfrecord']).map(lambda x: x)
        self._val_data = tf.data.TFRecordDataset(['data_validate.tfrecord'])

        self._nn = self._create_nn(activation)

        self._loss_function = tf.reduce_sum(tf.squared_difference(self._nn, self._Y))
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        self._train_op = optimizer.minimize(self._loss_function)

    def _create_nn(self, activation):
        l1 = tf.layers.dense(self._X, self.NEURONS[1], activation=activation)
        l2 = tf.layers.dense(l1, self.NEURONS[2], activation=activation)
        out = tf.layers.dense(l2, self.NEURONS[3], activation=tf.nn.softmax)
        return out

    def _run_example(self, sess: tf.Session, func: tf.Tensor, example: tf.Tensor):
        parsed = tf.parse_single_example(example, self._feature)

        input_str = sess.run(parsed['image/data']).values
        input = sess.run(tf.cast(tf.decode_raw(input_str, tf.uint8), tf.float32))
        output = sess.run(parsed['image/class'])
        output_arr = [[0 for i in range(51) if i != output]]

        return sess.run(func, {self._X: input, self._Y: output_arr})

    def _run_train(self, sess: tf.Session):
        train_it = self._train_data.make_one_shot_iterator()
        next_train = train_it.get_next()

        run = 0
        try:
            while True:
                run += 1
                self._run_example(sess, self._train_op, next_train)
        except tf.errors.OutOfRangeError:
            pass

    def _run_eval(self, sess: tf.Session):
        eval_it = self._val_data.make_one_shot_iterator()
        next_eval = eval_it.get_next()

        total_loss = 0.
        run = 0
        try:
            while True:
                run += 1
                loss = self._run_example(sess, self._loss_function, next_eval)
                total_loss += loss
        except tf.errors.OutOfRangeError:
            pass

        print("average: " + str(total_loss/run))

    def eval(self, sess: tf.Session):
        self._run_eval(sess)

    def train(self, sess: tf.Session, epochs, print_and_eval=True):
        for i in range(epochs):
            self._run_train(sess)

            if print_and_eval:
                print("after epoch: " + str(i + 1))
                self._run_eval(sess)

