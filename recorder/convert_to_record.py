from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import h5py
from workproperty import roi_property


EXP_TYPE_STR = roi_property.EXP_TYPE_STR[0]
EXP_NAME_STR = roi_property.EXP_NAME_STR[0]
DAT_TYPE_STR = roi_property.DAT_TYPE_STR[0]
SUB_STR = roi_property.SUB_STR[0]
CHAN_STR = roi_property.CHAN_STR

EEG_DATA = EXP_TYPE_STR + '_' + \
           EXP_NAME_STR + '_' + \
           SUB_STR + '_' + \
           DAT_TYPE_STR + '_' + \
           CHAN_STR
EEG_DATA_DIR = roi_property.FILE_DIR + \
               'rsvp_data/mat/' + EEG_DATA
EEG_TF_DIR = roi_property.FILE_DIR + \
               'rsvp_data/' + EEG_DATA
EEG_DATA_MAT = EEG_DATA_DIR + '.mat'

tf.app.flags.DEFINE_string('directory', roi_property.FILE_DIR+'rsvp_data/',
                           'Directory to download data files and write the '
                           'converted result')
tf.app.flags.DEFINE_integer('validation_size', 1000,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


def main(argv):
    # Get the data.
    f = h5py.File(EEG_DATA_MAT)
    train_x = f['train_x'][:]
    train_y = f['train_y'][:]
    test_x = f['test_x'][:]
    test_y = f['test_y'][:]

    # Extract it into numpy arrays.
    train_images = np.transpose(train_x, [0, 2, 3, 1])
    train_labels = train_y[:, 0]
    test_images = np.transpose(test_x, [0, 2, 3, 1])
    test_labels = test_y[:, 0]

    # For debugging check
    print('Training data shape: \t%s' % (train_images.shape,))
    print('Training label shape:\t%s' % (train_labels.shape,))
    print('Testing data shape:  \t%s' % (test_images.shape,))
    print('Testing label shape: \t%s' % (test_labels.shape,))
    print('max label value is: %s' % np.max(train_labels))
    print('min label value is: %s' % np.min(train_labels))

    # Generate a validation set.
    validation_images = train_images[:FLAGS.validation_size, :, :, :]
    validation_labels = train_labels[:FLAGS.validation_size]
    train_images = train_images[FLAGS.validation_size:, :, :, :]
    train_labels = train_labels[FLAGS.validation_size:]

    # Convert to Examples and write the result to TFRecords.
    convert_to(train_images, train_labels, EEG_TF_DIR+'.train')
    convert_to(validation_images, validation_labels, EEG_TF_DIR+'.validation')
    convert_to(test_images, test_labels, EEG_TF_DIR+'.test')


if __name__ == '__main__':
    tf.app.run()