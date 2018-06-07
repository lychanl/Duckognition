import tensorflow as tf
import operator
from tensorflow.contrib.opt import ScipyOptimizerInterface

class SVM:
    CLASSES = 50

    class _SVMSingleClass:

        class SupportVector:
            def __init__(self, input_size):
                self.value = tf.Variable(tf.random_normal(shape=(input_size,)))

        def __init__(self, input, input_size, support_vector_n):
            self.support_vectors = [self.SupportVector(input_size) for _ in range(support_vector_n)]
            self.bias = tf.Variable(tf.random_normal(shape=(1,)))
            self.neg_gamma = tf.Variable(-1.)
            self.weight = tf.Variable(tf.random_normal(shape=(1, support_vector_n)))
            self.kernels = tf.stack([
                        tf.exp(tf.multiply(
                            tf.norm(tf.subtract(input, vec.value), axis=1),
                            self.neg_gamma
                        ))
                        for vec in self.support_vectors
                    ], axis=1)
            self.output = tf.reshape(
                tf.subtract(tf.matmul(self.kernels, self.weight, transpose_b=True), self.bias),
                [-1])

            pass

        def trainable_vars(self):
            ret = [v.value for v in self.support_vectors]
            ret.append(self.bias)
            ret.append(self.neg_gamma)
            ret.append(self.weight)
            return ret

        def equality(self):
            return tf.reduce_sum(self.weight)

        def inequality(self):
            return -self.neg_gamma

        def create_optimizer(self, desired_output):
            return ScipyOptimizerInterface(
                tf.squared_difference(self.output, desired_output),
                self.trainable_vars(),
                equalities=[self.equality()],
                inequalities=[self.inequality()],
                method='SLSQP')

    def __init__(self, conv, support_vector_n):
        self._conv = conv

        conv_descr = tf.reshape(conv.descriptor, (-1, conv.descr_size))

        self.class_svms = [
            self._SVMSingleClass(conv_descr, conv.descr_size, support_vector_n)
            for _ in range(self.CLASSES)]

        self.trainable_vars = [var for vec in self.class_svms for var in vec.trainable_vars()]

        self.Y = tf.placeholder(tf.float32, [None, self.CLASSES])
        y1d = tf.reshape(self.Y, [self.CLASSES])

        self.output = tf.stack([v.output for v in self.class_svms], axis=1)

        self.loss = tf.reduce_sum(tf.squared_difference(self.output, self.Y), axis=-1)

        equalities = [v.equality() for v in self.class_svms]
        inequalities = [v.inequality() for v in self.class_svms]

        self.train_ops = [
            self.class_svms[i].create_optimizer(tf.slice(y1d, [i], [1]))
            for i in range(len(self.class_svms))]

    def varaibles(self):
        return self.trainable_vars

    def _run_example(self, sess: tf.Session, func, example: tf.Tensor, train=False):
        parsed = sess.run(example)

        input = parsed['image/data'].values
        output = parsed['image/class']

        if train:
            for train_op in self.train_ops:
                train_op.minimize(sess, feed_dict={self._conv.X: input, self.Y: output})

        ret, res = sess.run((func, self.output), {self._conv.X: input, self.Y: output})

        desired_el = [max(enumerate(o), key=operator.itemgetter(1))[0] for o in output]
        res_el = [max(enumerate(r), key=operator.itemgetter(1))[0] for r in res]
        #        print(res)
        #        print(desired)

        return ret, [d == r for d, r in zip(desired_el, res_el)]

    def _run_train(self, sess: tf.Session):
        sess.run(self._conv.train_it_initializer)

        run = 0
        total_loss = 0.
        correct_n = 0

        grads = []

        try:
            while True:
                loss, correct = self._run_example(sess, self.loss, self._conv.train_it_next, True)

                for l, c in zip(loss, correct):
                    total_loss += l
                    run += 1
                    if c:
                        correct_n += 1

        except tf.errors.OutOfRangeError:
            pass

        print("train average loss: " + str(total_loss / run) + " correct predictions: " + str(correct_n / run))

    def _run_eval(self, sess: tf.Session):
        sess.run(self._conv.eval_it_initializer)

        total_loss = 0.
        run = 0
        correct_n = 0
        try:
            while True:
                run += 1
                loss, correct = self._run_example(sess, self.loss, self._conv.eval_it_next)

                for l, c in zip(loss, correct):
                    total_loss += l
                    run += 1
                    if c:
                        correct_n += 1

        except tf.errors.OutOfRangeError:
            pass

        print("eval average: " + str(total_loss / run) + " correct predictions: " + str(correct_n / run))

    def eval(self, sess: tf.Session):
        self._run_eval(sess)

    def train(self, sess: tf.Session, epochs, print_and_eval=True):
        for i in range(int(epochs)):
            self._run_train(sess)

            if print_and_eval:
                print("after epoch: " + str(i + 1))
                self._run_eval(sess)

