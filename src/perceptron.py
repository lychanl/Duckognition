import tensorflow as tf
import operator


class Perceptron:

    INPUTS_FILTERED = 20*20*12
    INPUTS_UNFILTERED = 20*20*3
    CLASSES = 50
    HIDDEN_NEURONS = (150, 100)

    _feature = {
        'image/data': tf.VarLenFeature(tf.string),
        'image/class': tf.FixedLenFeature([], tf.int64),
    }

    def __init__(self, learning_rate, momentum, activation, filtered):
        self._filtered = filtered
        self._inputs = self.INPUTS_FILTERED if filtered else self.INPUTS_UNFILTERED

        self._X = tf.placeholder(tf.float32, [None, self._inputs])
        self._Y = tf.placeholder(tf.float32, [None, self.CLASSES])
        self._train_data = tf.data.TFRecordDataset(['data_train.tfrecord' if filtered
                                                    else 'data_train_unf.tfrecord'])
        self._val_data = tf.data.TFRecordDataset(['data_validate.tfrecord' if filtered
                                                  else 'data_validate_unf.tfrecord'])

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
        hidden = tf.layers.dense(self._X, self.HIDDEN_NEURONS[0], activation=activation)
        hidden2 = tf.layers.dense(hidden, self.HIDDEN_NEURONS[1], activation=activation)
        out = tf.layers.dense(hidden2, self.CLASSES, activation=tf.nn.softmax)
        return out

    def _run_example(self, sess: tf.Session, func, example: tf.Tensor):
        parsed = sess.run(example)

        input_str = parsed['image/data'].values
        input = sess.run(self._input_decoder, {self._input_to_decode: input_str})
        input = [[input[0][i] for i in range(len(input[0])) if (i % 3 == 0 or not self._filtered)]]
        output = parsed['image/class']
        output_arr = [[0 if i != output else 1 for i in range(50)]]

        ret, res, desired = sess.run((func, self._nn, self._Y), {self._X: input, self._Y: output_arr})

        desired_el, _ = max(enumerate(desired[0]), key=operator.itemgetter(1))
        res_el, _ = max(enumerate(res[0]), key=operator.itemgetter(1))
#        print(res)
#        print(desired)

        return ret, desired_el == res_el

    def _run_train(self, sess: tf.Session):
        sess.run(self._train_it_initializer)

        run = 0
        total_loss = 0.
        correct_n = 0
        try:
            while True:
                run += 1
                (loss, _), correct = self._run_example(sess, (self._loss_function, self._train_op), self._train_it_next)
                total_loss += loss

                if correct:
                    correct_n += 1
        except tf.errors.OutOfRangeError:
            pass

        print("train average loss: " + str(total_loss / run) + " correct predictions: " + str(correct_n / run))

    def _run_eval(self, sess: tf.Session):
        sess.run(self._eval_it_initializer)

        total_loss = 0.
        run = 0
        correct_n = 0
        try:
            while True:
                run += 1
                loss, correct = self._run_example(sess, self._loss_function, self._eval_it_next)
                total_loss += loss

                if correct:
                    correct_n += 1
        except tf.errors.OutOfRangeError:
            pass

        print("eval average: " + str(total_loss/run) + " correct predictions: " + str(correct_n / run))

    def eval(self, sess: tf.Session):
        self._run_eval(sess)

    def train(self, sess: tf.Session, epochs, print_and_eval=True):
        for i in range(epochs):
            self._run_train(sess)

            if print_and_eval:
                print("after epoch: " + str(i + 1))
                self._run_eval(sess)

