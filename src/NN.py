import tensorflow as tf
import operator
import collections


class NN:

    CLASSES = 50

    _feature = {
        'image/data': tf.VarLenFeature(tf.string),
        'image/class': tf.FixedLenFeature([50], tf.int64),
    }

    def __init__(self, learning_rate, momentum, tfrecord_suffix, width, height, channels, cross_entropy=None, train_batch_size=5000):

        self.X = tf.placeholder(tf.string)
        self.Y = tf.placeholder(tf.float32, [None, self.CLASSES])
        self.input = tf.reshape(tf.cast(tf.decode_raw(self.X, tf.uint8), tf.float32), [-1, width, height, channels])  # decode input

        train_data = tf.data.TFRecordDataset(['data_train' + tfrecord_suffix + '.tfrecord'])
        val_data = tf.data.TFRecordDataset(['data_validate' + tfrecord_suffix + '.tfrecord'])

        train_it = train_data.shuffle(10000).batch(train_batch_size).make_initializable_iterator()
        eval_it = val_data.shuffle(10000).batch(5000).make_initializable_iterator()

        self.train_it_next = tf.parse_example(train_it.get_next(), self._feature)
        self.train_it_initializer = train_it.initializer

        self.eval_it_next = tf.parse_example(eval_it.get_next(), self._feature)
        self.eval_it_initializer = eval_it.initializer

        self._nn = self._create_nn()

        self._loss_function = cross_entropy(self._nn, self.Y) if cross_entropy is not None \
            else tf.reduce_sum(tf.squared_difference(self._nn, self.Y), 1)

        optimizer = tf.train.AdamOptimizer()  # (learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
        self._train_op = optimizer.minimize(self._loss_function)

    def _create_nn(self):
        raise NotImplementedError()

    def _run_example(self, sess: tf.Session, func, example: tf.Tensor):
        parsed = sess.run(example)

        input = parsed['image/data'].values
        output = parsed['image/class']

        ret, res = sess.run((func, self._nn), {self.X: input, self.Y: output})

        desired_el = [max(enumerate(o), key=operator.itemgetter(1))[0] for o in output]
        res_el = [max(enumerate(r), key=operator.itemgetter(1))[0] for r in res]
#        print(res)
#        print(desired)

        return ret, [d == r for d, r in zip(desired_el, res_el)]

    def _run_train(self, sess: tf.Session):
        sess.run(self.train_it_initializer)

        run = 0
        total_loss = 0.
        correct_n = 0

        grads = []

        try:
            while True:
                (loss, _), correct = self._run_example(sess, (self._loss_function, self._train_op), self.train_it_next)

                if loss is collections.Iterable:
                    for l, c in zip(loss, correct):
                        total_loss += l
                        run += 1
                        if c:
                            correct_n += 1
                else:
                    total_loss += loss
                    for c in correct:
                        run += 1
                        if c:
                            correct_n += 1


        except tf.errors.OutOfRangeError:
            pass

        print("train average loss: " + str(total_loss / run) + " correct predictions: " + str(correct_n / run))

    def _run_eval(self, sess: tf.Session):
        sess.run(self.eval_it_initializer)

        total_loss = 0.
        run = 0
        correct_n = 0
        try:
            while True:
                run += 1
                loss, correct = self._run_example(sess, self._loss_function, self.eval_it_next)

                if loss is collections.Iterable:
                    for l, c in zip(loss, correct):
                        total_loss += l
                        run += 1
                        if c:
                            correct_n += 1
                else:
                    total_loss += loss
                    for c in correct:
                        run += 1
                        if c:
                            correct_n += 1


        except tf.errors.OutOfRangeError:
            pass

        print("eval average: " + str(total_loss/run) + " correct predictions: " + str(correct_n / run))

    def eval(self, sess: tf.Session):
        self._run_eval(sess)

    def train(self, sess: tf.Session, epochs, print_and_eval=True):
        for i in range(int(epochs)):
            self._run_train(sess)

            if print_and_eval:
                print("after epoch: " + str(i + 1))
                self._run_eval(sess)

