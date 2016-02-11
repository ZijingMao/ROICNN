# Created by Zijing Mao at 2/10/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from workproperty import roi_property
from numpy import random
import tensorflow as tf
import numpy as np
import biosemi_chan_mapper
import unittest
import split_eeg
import concat_eeg


EEG_SIGNAL_SIZE = roi_property.EEG_SIGNAL_SIZE
FAKE_EEG_SIGNAL = random.randint(0, 10, size=(10, EEG_SIGNAL_SIZE, EEG_SIGNAL_SIZE, 1))
FAKE_EEG_SIGNAL = tf.constant(FAKE_EEG_SIGNAL, dtype=np.float32)


class TestImageMapper(unittest.TestCase):

    sess = tf.InteractiveSession()

    def test_biosemi_conv_mapper(self):
        for chan_num in roi_property.CONV_CHAN_INFO:
            conv_256_idx = biosemi_chan_mapper.conv_mapper(chan_num)
            self.assertEqual(conv_256_idx.shape, (chan_num, roi_property.BIOSEMI_CONV))

    def test_biosemi_pool_mapper(self):
        for chan_num in roi_property.POOL_CHAN_INFO:
            pool_256_idx = biosemi_chan_mapper.pool_mapper(chan_num)
            self.assertEqual(pool_256_idx.shape, (chan_num, roi_property.BIOSEMI_POOL))

    def test_biosemi_pool_all_mapper(self):
        for chan_num in roi_property.POOL_CHAN_INFO:
            pool_256_idx = biosemi_chan_mapper.pool_mapper(chan_num, roi_property.BIOSEMI_POOL_ALL)
            self.assertEqual(pool_256_idx.shape, (chan_num, roi_property.BIOSEMI_POOL_ALL))

    def test_replace_mapper(self):
        for chan_num in roi_property.CONV_CHAN_INFO:
            conv_256_idx = biosemi_chan_mapper.conv_mapper(chan_num)
            conv_256_idx = biosemi_chan_mapper.replace_mapper_idx(conv_256_idx)
            self.assertEqual(conv_256_idx.shape, (chan_num, roi_property.BIOSEMI_CONV))

    def test_split_eeg(self):
        fake_image_shape = FAKE_EEG_SIGNAL.get_shape().as_list()
        fake_image_list = split_eeg.split_eeg_signal(FAKE_EEG_SIGNAL)

        # check if the split is a list
        self.assertIsInstance(fake_image_list, list)

        # check the size of the list is equal to the split size
        self.assertEqual(len(fake_image_list), roi_property.EEG_SIGNAL_SIZE)

        fake_image_shape[1] = 1
        # check all the split have the desired dimension
        for fake_image in fake_image_list:
            self.assertEqual(fake_image.get_shape().as_list(), fake_image_shape)

    def test_conv_channel(self):
        fake_image_list = split_eeg.split_eeg_signal_axes(FAKE_EEG_SIGNAL, 1)
        kernel_tensor, input_shape = \
            concat_eeg.conv_eeg_signal_channel(fake_image_list, 256, 1)
        # check the size of the shape is equal to the concat size
        self.assertEqual(kernel_tensor.get_shape().as_list(), input_shape)

    def test_conv_time(self):
        fake_image_list = split_eeg.split_eeg_signal_axes(FAKE_EEG_SIGNAL, 2)
        kernel_tensor, input_shape = \
            concat_eeg.conv_eeg_signal_time(fake_image_list, np.arange(0, 256))
        # check the size of the shape is equal to the concat size
        self.assertEqual(kernel_tensor.get_shape().as_list(), [10, 256, 256*5, 1])
        self.assertEqual(input_shape, [10, 256, 256*5, 1])

    def test_conv_channel_time(self):
        # first split the eeg signal of (10, 256, 256, 1) to 256 * (10, 1, 256, 1)
        fake_image_list = split_eeg.split_eeg_signal_axes(FAKE_EEG_SIGNAL, 1)
        # then concat the eeg signal use 256 channel kernel index to (10, 256*5, 256, 1)
        kernel_tensor, input_shape = \
            concat_eeg.conv_eeg_signal_channel(fake_image_list, 256, 1)
        self.assertEqual(kernel_tensor.get_shape().as_list(), [10, 256*5, 256, 1])
        self.assertEqual(input_shape, [10, 256*5, 256, 1])

        # then split the new eeg signal of (10, 1280, 256, 1) to 256 * (10, 1280, 1, 1)
        fake_image_list = split_eeg.split_eeg_signal_axes(kernel_tensor, 2)
        # then concat the eeg signal use 256 channel kernel index to (10, 256*5, 256*5, 1)
        kernel_tensor, input_shape = \
            concat_eeg.conv_eeg_signal_time(fake_image_list, np.arange(0, 256))

        # check the size of the shape is equal to the concat size
        self.assertEqual(kernel_tensor.get_shape().as_list(), [10, 256*5, 256*5, 1])
        self.assertEqual(input_shape, [10, 256*5, 256*5, 1])

    def test_pool_channel_time(self):
        # first split the eeg signal of (10, 256, 256, 1) to 256 * (10, 1, 256, 1)
        fake_image_list = split_eeg.split_eeg_signal_axes(FAKE_EEG_SIGNAL, 1)
        # then concat the eeg signal use 256 channel kernel index to (10, 128*2, 256, 1)
        kernel_tensor, input_shape = \
            concat_eeg.pool_eeg_signal_channel(fake_image_list, 128, 1)
        self.assertEqual(kernel_tensor.get_shape().as_list(), [10, 128*2, 256, 1])
        self.assertEqual(input_shape, [10, 128*2, 256, 1])

    sess.close()

if __name__ == '__main__':
    unittest.main()
