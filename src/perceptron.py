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
        self._train_data = tf.data.TFRecordDataset(['data_train.tfrecord'])
        self._val_data = tf.data.TFRecordDataset(['data_validate.tfrecord'])

        train_it = self._train_data.make_initializable_iterator()
        eval_it = self._val_data.make_initializable_iterator()

        self._train_it_next = tf.parse_single_example(eval_it.get_next(), self._feature)
        self._train_it_initializer = eval_it.initializer

        self._eval_it_next = tf.parse_single_example(train_it.get_next(), self._feature)
        self._eval_it_initializer = train_it.initializer

        self._nn = self._create_nn(activation)

        self._loss_function = tf.reduce_sum(tf.squared_difference(self._nn, self._Y))
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        self._train_op = optimizer.minimize(self._loss_function)

        self._input_to_decode = tf.placeholder(tf.string)
        self._input_decoder = tf.cast(tf.decode_raw(self._input_to_decode, tf.uint8), tf.float32)

    def _create_nn(self, activation):
        l1 = tf.layers.dense(self._X, self.NEURONS[1], activation=activation)
        l2 = tf.layers.dense(l1, self.NEURONS[2], activation=activation)
        out = tf.layers.dense(l2, self.NEURONS[3], activation=tf.nn.softmax)
        return out

    def _run_example(self, sess: tf.Session, func: tf.Tensor, example: tf.Tensor):
        parsed = sess.run(example)

        input_str = parsed['image/data'].values
        input = sess.run(self._input_decoder, {self._input_to_decode: input_str})
        output = parsed['image/class']
        output_arr = [[0 for i in range(51) if i != output]]

        return sess.run(func, {self._X: input, self._Y: output_arr})

    def _run_train(self, sess: tf.Session):
        sess.run(self._train_it_initializer)

        run = 0
        try:
            while True:
                run += 1
                self._run_example(sess, self._train_op, self._train_it_next)
        except tf.errors.OutOfRangeError:
            pass

    def _run_eval(self, sess: tf.Session):
        sess.run(self._eval_it_initializer)

        total_loss = 0.
        run = 0
        try:
            while True:
                run += 1
                loss = self._run_example(sess, self._loss_function, self._eval_it_next)
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

