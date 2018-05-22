import tensorflow as tf
import operator


class NN:

    INPUTS_FILTERED = 20*20*12
    INPUTS_UNFILTERED = 20*20*3
    CLASSES = 50

    _feature = {
        'image/data': tf.VarLenFeature(tf.string),
        'image/class': tf.FixedLenFeature([1, 50], tf.int64),
    }

    def __init__(self, learning_rate, momentum, activation, filtered):
        self._filtered = filtered
        self._inputs = self.INPUTS_FILTERED if filtered else self.INPUTS_UNFILTERED

        self.X = tf.placeholder(tf.string)
        self.Y = tf.placeholder(tf.float32, [None, self.CLASSES])
        self.input = tf.reshape(tf.cast(tf.decode_raw(self.X, tf.uint8), tf.float32), [1, self._inputs])  # decode input

        train_data = tf.data.TFRecordDataset(['data_train_proc.tfrecord' if filtered
                                                    else 'data_train_unf_proc.tfrecord'])
        val_data = tf.data.TFRecordDataset(['data_validate_proc.tfrecord' if filtered
                                                  else 'data_validate_unf_proc.tfrecord'])

        train_it = train_data.shuffle(10000).make_initializable_iterator()
        eval_it = val_data.shuffle(10000).make_initializable_iterator()

        self._train_it_next = tf.parse_single_example(train_it.get_next(), self._feature)
        self._train_it_initializer = train_it.initializer

        self._eval_it_next = tf.parse_single_example(eval_it.get_next(), self._feature)
        self._eval_it_initializer = eval_it.initializer

        self._nn = self._create_nn(activation)

        self._loss_function = tf.reduce_sum(tf.squared_difference(self._nn, self.Y))
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        self._train_op = optimizer.minimize(self._loss_function)

    def _create_nn(self, activation):
        raise NotImplementedError()

    def _run_example(self, sess: tf.Session, func, example: tf.Tensor):
        parsed = sess.run(example)

        input = parsed['image/data'].values
        output = parsed['image/class']

        ret, res = sess.run((func, self._nn), {self.X: input, self.Y: output})

        desired_el, _ = max(enumerate(output[0]), key=operator.itemgetter(1))
        res_el, _ = max(enumerate(res[0]), key=operator.itemgetter(1))
#        print(res)
#        print(desired)

        return ret, desired_el == res_el

    def _run_train(self, sess: tf.Session):
        sess.run(self._train_it_initializer)

        run = 0
        total_loss = 0.
        correct_n = 0

        grads = []

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

