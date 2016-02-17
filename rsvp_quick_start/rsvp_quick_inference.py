# -*- coding: utf-8 -*-
"""
Created on 2/17/16 3:29 PM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow.python.platform
import tensorflow as tf
from workproperty import roi_property

# The RSVP dataset has 2 classes, representing the digits 0 through 1.
NUM_CLASSES = roi_property.BINARY_LABEL
IMAGE_SIZE = roi_property.EEG_SIGNAL_SIZE

flags = tf.app.flags
FLAGS = flags.FLAGS

# region define the variable methods


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# endregion


# region define 1-layer models here

def inference_local_st_filter(images, keep_prob):

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 64, 1, 8],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv1 = tf.nn.relu(bias, name=scope.name)


    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in conv1.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(conv1, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)

    # dropout1
    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(local3, keep_prob)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[128, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(dropout1, weights, biases, name=scope.name)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [128, NUM_CLASSES],
                                              stddev=1/128.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        logits = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)

    return logits


def inference_temporal_filter(images, keep_prob):

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 64, 1, 8],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv1 = tf.nn.relu(bias, name=scope.name)


    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in conv1.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(conv1, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)

    # dropout1
    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(local3, keep_prob)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[128, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(dropout1, weights, biases, name=scope.name)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [128, NUM_CLASSES],
                                              stddev=1/128.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        logits = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)

    return logits


def inference_global_st_filter(images, keep_prob):

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[64, 4, 1, 4],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [4], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv1 = tf.nn.relu(bias, name=scope.name)


    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in conv1.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(conv1, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)

    # dropout1
    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(local3, keep_prob)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[128, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(dropout1, weights, biases, name=scope.name)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [128, NUM_CLASSES],
                                              stddev=1/128.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        logits = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)

    return logits


def inference_spatial_filter(images, keep_prob):

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[64, 1, 1, 4],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [4], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv1 = tf.nn.relu(bias, name=scope.name)


    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in conv1.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(conv1, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)

    # dropout1
    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(local3, keep_prob)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[128, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(dropout1, weights, biases, name=scope.name)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [128, NUM_CLASSES],
                                              stddev=1/128.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        logits = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)

    return logits

# endregion


# region define 2-layer models

# endregion
