# Created by Zijing Mao at 2/10/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from workproperty import roi_property
from numpy import random
import numpy as np
import tensorflow as tf
import biosemi_chan_mapper


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

    sess.close()

if __name__ == '__main__':
    unittest.main()