# Created by Zijing Mao at 2/13/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tensorflow.python.platform
import h5py
import numpy as np
from workproperty import roi_property
import tensorflow as tf


def dense_to_one_hot(labels_dense, num_classes=roi_property.BINARY_LABEL):
    '''
    Convert class labels from scalars to one-hot vectors.
    Args:
        labels_dense: the label of 1 vector
        num_classes: the number of classes

    Returns: the one hot label matrix

    '''
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    ravel_index = index_offset + labels_dense.ravel()
    labels_one_hot.flat[ravel_index] = 1
    return labels_one_hot


class DataSet(object):

    def __init__(self, images, labels, fake_data=False, one_hot=False,
                 dtype=tf.float32, reshape_tensor=True):
        """Construct a DataSet.

        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # assert images.shape[3] == 1
            if reshape_tensor:
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_data,
                   fake_data=False,
                   one_hot=False,
                   dtype=tf.float32,
                   scale_data=False,
                   reshape_t=True,
                   validation_size=roi_property.MEDIUM_VALID_SIZE):
    class DataSets(object):
        pass
    data_sets = DataSets()

    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
        data_sets.train = fake()
        data_sets.validation = fake()
        data_sets.test = fake()
        return data_sets

    # Get the data.
    f = h5py.File(train_data)
    train_x = f['train_x'][:]
    train_y = f['train_y'][:]
    test_x = f['test_x'][:]
    test_y = f['test_y'][:]

    # Extract it into numpy arrays.
    train_images = np.transpose(train_x, [0, 2, 3, 1])
    # train_images = np.transpose(train_x, [0, 3, 2, 1])
    train_labels = train_y[:, 0].astype(int)
    test_images = np.transpose(test_x, [0, 2, 3, 1])
    # test_images = np.transpose(test_x, [0, 3, 2, 1])
    test_labels = test_y[:, 0].astype(int)

    if one_hot:
        train_labels = dense_to_one_hot(train_labels)
        test_labels = dense_to_one_hot(test_labels)

    # For debugging check
    print('Training data shape: \t%s' % (train_images.shape,))
    print('Training label shape:\t%s' % (train_labels.shape,))
    print('Testing data shape:  \t%s' % (test_images.shape,))
    print('Testing label shape: \t%s' % (test_labels.shape,))
    max_image_value = np.max(train_images)
    min_image_value = np.min(train_images)
    print('max input value is: %s' % max_image_value)
    print('min input value is: %s' % min_image_value)

    # TODO random permute once for training
    rand_position = np.random.permutation(len(train_labels))
    train_labels = train_labels[rand_position]
    train_images = train_images[rand_position, :, :, :]
    print('random permuted training data')

    if scale_data:
        scale_range = 1.0 / (max_image_value-min_image_value)
        sub_train_images = np.subtract(train_images, min_image_value)
        sub_train_images = sub_train_images.astype(np.float32)
        train_images = np.multiply(sub_train_images, scale_range)

        sub_test_images = np.subtract(test_images, min_image_value)
        sub_test_images = sub_test_images.astype(np.float32)
        test_images = np.multiply(sub_test_images, scale_range)

        max_image_value = np.max(train_images)
        min_image_value = np.min(train_images)
        print('Rescale to: [%s, %s]' % (min_image_value, max_image_value))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    data_sets.train = DataSet(train_images,
                              train_labels,
                              dtype=dtype,
                              reshape_tensor=reshape_t)
    data_sets.validation = DataSet(validation_images,
                                   validation_labels,
                                   dtype=dtype,
                                   reshape_tensor=reshape_t)
    data_sets.test = DataSet(test_images,
                             test_labels,
                             dtype=dtype,
                             reshape_tensor=reshape_t)
    data_sets.feature_shape = train_images.shape

    return data_sets
