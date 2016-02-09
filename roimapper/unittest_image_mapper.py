# Created by Zijing Mao at 2/8/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import concat_image
import split_image
from workproperty import roi_property
from numpy import random
import numpy as np
import tensorflow as tf


DIGIT_IMAGE_SIZE = roi_property.DIGIT_IMAGE_SIZE
FAKE_DIGIT_IMAGE = random.randint(0, 10, size=(10, DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1))
FAKE_DIGIT_IMAGE = tf.constant(FAKE_DIGIT_IMAGE, dtype=np.float32)


class TestImageMapper(unittest.TestCase):

    sess = tf.InteractiveSession()

    def test_split(self):
        fake_image_shape = FAKE_DIGIT_IMAGE.get_shape().as_list()
        fake_image_list = split_image.split_digit_image(FAKE_DIGIT_IMAGE)

        # check if the split is a list
        self.assertIsInstance(fake_image_list, list)

        # check the size of the list is equal to the split size
        self.assertEqual(len(fake_image_list), roi_property.DIGIT_IMAGE_SIZE)

        fake_image_shape[1] = 1
        # check all the split have the desired dimension
        for fake_image in fake_image_list:
            self.assertEqual(fake_image.get_shape().as_list(), fake_image_shape)

    def test_concat(self):
        fake_image_list = split_image.split_digit_image(FAKE_DIGIT_IMAGE)
        kernel_tensor, input_shape = \
            concat_image.concat_digit_image(fake_image_list, np.arange(0, 28), 5, 1, 3)
        # check the size of the shape is equal to the concat size
        self.assertEqual(kernel_tensor.get_shape().as_list(), input_shape)

    sess.close()

if __name__ == '__main__':
    unittest.main()