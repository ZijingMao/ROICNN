# -*- coding: utf-8 -*-
"""
Created on 2/17/16 3:29 PM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from workproperty import roi_property
from roimapper import concat_eeg
from roimapper import split_eeg
import numpy as np

import tensorflow as tf

# The RSVP dataset has 2 classes, representing the digits 0 through 1.
NUM_CLASSES = roi_property.BINARY_LABEL  # replace with multiple labels
IMAGE_SIZE = roi_property.EEG_SIGNAL_SIZE
KERNEL_SIZE = roi_property.BIOSEMI_CONV

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


def _print_tensor_size(given_tensor):
    '''

    Args:
        given_tensor: the tensor want to see detail information

    Returns:
        print out the tensor shape ar other information as needed

    '''
    # print the shape of tensor
    print("="*78)
    print("Tensor Name: " + given_tensor.name)
    tensor_shape = given_tensor.get_shape()
    print(tensor_shape.as_list())


# endregion

# TODO try catch if the shape does not match, if the augment is not a tensor
# region define the eeg mappers for conv & pool

def inference_augment_st_filter(images):
    # augment on spatial domain
    augment = inference_augment_s_filter(images)

    # augment on temporal domain
    split_dim = 2   # split the temporal domain and rearrange them to repeat time domain
    input_image_list = split_eeg.split_eeg_signal_axes(augment,
                                                       split_dim=split_dim)  # 2 represents temporal domain
    input_image_length = len(input_image_list)
    augment, _ = concat_eeg.conv_eeg_signal_time(input_image_list,
                                                 np.arange(0, input_image_length),
                                                 KERNEL_SIZE, 2)
    _print_tensor_size(augment)

    return augment


def inference_augment_s_rep_t_filter(images):
    # augment on spatial domain
    augment = inference_augment_s_filter(images)

    # augment on temporal domain
    split_dim = 2   # split the temporal domain and rearrange them to repeat time domain
    input_image_list = split_eeg.split_eeg_signal_axes(augment,
                                                       split_dim=split_dim)  # 2 represents temporal domain
    input_image_length = len(input_image_list)
    augment, _ = concat_eeg.conv_eeg_signal_time(input_image_list,
                                                 np.arange(0, input_image_length),
                                                 KERNEL_SIZE, 2, is_rep=True)
    _print_tensor_size(augment)

    return augment


def inference_augment_s_filter(images):
    # recommend to use
    # augment
    split_dim = 1   # 1 represents split on spatial domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)
    input_image_length = len(input_image_list)
    augment, _ = concat_eeg.conv_eeg_signal_channel(input_image_list, input_image_length, 1)
    _print_tensor_size(augment)

    return augment


def inference_pooling_s_filter(images, kheight=2, kwidth=2):
    # channel domain pooling mapper
    split_dim = 1   # 1 represents split on spatial domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)
    input_image_length = len(input_image_list)
    # the pooling mapper should choose half size of the image size
    pool_s, _ = concat_eeg.pool_eeg_signal_channel(input_image_list, input_image_length/2, 1)
    _print_tensor_size(pool_s)

    # apply the normal max pooling methods with stride = 2
    pool_s = inference_pooling_n_filter(pool_s, kheight, kwidth)

    return pool_s


def inference_pooling_t_filter(images, kheight=2, kwidth=2):
    # channel domain pooling mapper
    split_dim = 1   # 1 represents split on spatial domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)
    input_image_length = len(input_image_list)
    # the pooling mapper should choose half size of the image size
    pool_t, _ = concat_eeg.pool_eeg_signal_channel(input_image_list, input_image_length/2, 1, is_rep=True)
    _print_tensor_size(pool_t)

    # apply the normal max pooling methods with stride = 2
    pool_t = inference_pooling_n_filter(pool_t, kheight, kwidth)

    return pool_t


def inference_pooling_n_filter(pool_s, kheight=2, kwidth=2):

    pool_s = tf.nn.max_pool(pool_s, ksize=[1, kheight, kwidth, 1],
                            strides=[1, kheight, kwidth, 1], padding='VALID')
    _print_tensor_size(pool_s)

    return pool_s


# endregion


# region define the fully connected layer

def inference_fully_connected_1layer(conv_output, keep_prob):

    # local1
    with tf.variable_scope('local1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in conv_output.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(conv_output, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        _print_tensor_size(local3)

    # dropout1
    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(local3, keep_prob)
        # _print_tensor_size(dropout1) # does not exist tensor shape

    # local2
    with tf.variable_scope('local2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[128, 128],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(dropout1, weights, biases, name=scope.name)
        _print_tensor_size(local4)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [128, NUM_CLASSES],
                                              stddev=1/128.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        logits = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)
        _print_tensor_size(logits)

    return logits

# endregion


# region define 1-layer modules here

def inference_local_st_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    augment = inference_augment_s_filter(images)

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(augment, kernel, [1, 5, 5, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_local_st5_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    augment = inference_augment_st_filter(images)

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(augment, kernel, [1, 5, 5, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_temporal_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 5, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_channel_wise_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    input_shape = images.get_shape().as_list()
    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, input_shape[2], in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_global_ts_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    input_shape = images.get_shape().as_list()
    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, input_shape[2], in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_global_st_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    input_shape = images.get_shape().as_list()
    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[input_shape[1], 5, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_roi_global_ts_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    augment = inference_augment_s_filter(images)

    input_shape = images.get_shape().as_list()
    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, input_shape[2], in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(augment, kernel, [1, 5, 5, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_roi_s_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    augment = inference_augment_s_rep_t_filter(images)

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(augment, kernel, [1, 5, 5, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_spatial_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 1, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_time_wise_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    input_shape = images.get_shape().as_list()
    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[input_shape[1], 1, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_1x1_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_5x5_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_depthwise_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.depthwise_conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_inception_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    # will be implement in the future
    # TODO 1x1 + 3x3 + 5x5 + pooling => next layer
    pass

# endregion


# region define 2-layer modules

def inference_separable_filter(images, conv_layer_scope, in_feat=1, mid_feat=2, out_feat=4):

    # conv_output = 5x5/1x1
    with tf.variable_scope(conv_layer_scope) as scope:
        # perform better when kernel size is small
        feat_multiplier = mid_feat
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, in_feat, feat_multiplier],
                                             stddev=1e-2, wd=0.0)
        kernel_pointwise = _variable_with_weight_decay('weights_point',
                                                       shape=[1, 1, feat_multiplier * in_feat, out_feat],
                                                       stddev=1e-2, wd=0.0)
        conv = tf.nn.separable_conv2d(images, kernel, kernel_pointwise, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def inference_residual_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    # will be implement in the future
    # TODO conv_output = filter(images, {W_i}) + [W_s]*images
    pass


def inference_inception2_filter(images, conv_layer_scope, in_feat=1, out_feat=4):
    
    # conv_output
    # will be implement in the future
    # TODO 1x1 + 1x1/3x3 + 1x1/5x5 + 3x3 pooling/1x1 => next layer
    pass

# endregion
