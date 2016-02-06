from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import tensorflow.python.platform
import tensorflow as tf

import numpy as np

SEED = 66478  # Set to None for random seed.


# conv1weights = np.random.randint(1, 5, size=(3, 3))
# conv1weights = np.array(conv1weights[None, :, :, None], dtype='f')
# conv1weightstensor = tf.constant(conv1weights)
# conv1weightstensor = tf.Variable(conv1weightstensor)
#
# datamapmatrix = np.random.randint(1, 5, size=(1, 1))
# datamapmatrix = np.array(datamapmatrix[:, :, None, None], dtype='f')
# datamaptensor = tf.constant(datamapmatrix)
# datamaptensor = tf.Variable(datamaptensor)

conv1weights = np.random.randint(1, 5, size=(1, 1, 1, 1))
conv1weights = np.array(conv1weights, dtype='f')
conv1weightstensor = tf.Variable(tf.constant(conv1weights, shape=[1,1,1,1]))
datamaptensor = tf.Variable(tf.random_normal([1,2,2,1]))
# conv1weightstensor = tf.Variable(tf.random_normal([1,1,1,1]))


conv = tf.nn.conv2d(datamaptensor,
                    conv1weightstensor,
                    strides=[1, 1, 1, 1],   # the stride of the height should change to 5
                    padding='SAME')

init = tf.initialize_all_variables()