import tensorflow as tf
import numpy as np


def int64_feature(value):
    """Returns a TF-Feature of int64s.
    Args:
      value: A scalar
    Returns:
      a TF-Feature.
    """
    values = [1 if i == value else 0 for i in range(50)]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(values)]))


def image_to_tfexample(image_data, class_id):

    return tf.train.Example(features=tf.train.Features(feature={
        'image/data': bytes_feature(image_data),
        'image/class': int64_feature(class_id),
    }))


input_feature = {
        'image/data': tf.VarLenFeature(tf.string),
        'image/class': tf.FixedLenFeature([], tf.int64),
    }


def run_example(sess: tf.Session, output_file, example, input_decoder, input_to_decode):
        parsed = sess.run(example)

        data_str = parsed['image/data'].values
#        data = [data_str[0][i] for i in range(len(data_str[0])) if (i % 3 == 0)]       # filtered
        data = data_str[0]                                                              # unfiltered
        img_class = parsed['image/class']
        data_str = bytearray()
        for e in data:
            data_str.append(e.to_bytes(1, 'little')[0])

        output_file.write(image_to_tfexample(data_str, img_class).SerializeToString())


def run_file(sess, input_name, output_name):
    with tf.python_io.TFRecordWriter(output_name) as tfrecord_writer:
        input = tf.data.TFRecordDataset([input_name])
        it = input.make_one_shot_iterator()

        it_next = tf.parse_single_example(it.get_next(), input_feature)

        input_to_decode = tf.placeholder(tf.string)
        input_decoder = tf.decode_raw(input_to_decode, tf.uint8)

        try:
            while True:
                run_example(sess, tfrecord_writer, it_next, input_decoder, input_to_decode)
        except tf.errors.OutOfRangeError:
            pass


with tf.Session() as sess:
    run_file(sess, "data_train_50x50.tfrecord", "data_train_50x50_proc.tfrecord")
    run_file(sess, "data_validate_50x50.tfrecord", "data_validate_50x50_proc.tfrecord")
