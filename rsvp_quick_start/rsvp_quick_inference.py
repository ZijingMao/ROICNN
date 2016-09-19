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

batch_size = roi_property.BATCH_SIZE
filterh = 4
filterw = 4

def set_batch_size(_batch_size):
    global batch_size
    batch_size = _batch_size

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
    #with tf.device('/cpu:0'):
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


def _uniforn_variable(name, shape, rng, wd):
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
                           tf.random_uniform_initializer(-rng, rng))
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

def inference_augment_t_filter(images, size):
    # augment on temporal domain
    split_dim = 2  # split the temporal domain and rearrange them to repeat time domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)  # 2 represents temporal domain
    input_image_length = len(input_image_list)
    augment, _ = concat_eeg.conv_eeg_signal_time(input_image_list,
                                                 np.arange(0, input_image_length),
                                                 size, 2)
    _print_tensor_size(augment)

    return augment


def inference_augment_st_filter(images, size):
    # augment on spatial domain
    augment = inference_augment_s_filter(images)

    # augment on temporal domain
    split_dim = 2   # split the temporal domain and rearrange them to repeat time domain
    input_image_list = split_eeg.split_eeg_signal_axes(augment,
                                                       split_dim=split_dim)  # 2 represents temporal domain
    input_image_length = len(input_image_list)
    augment, _ = concat_eeg.conv_eeg_signal_time(input_image_list,
                                                 np.arange(0, input_image_length),
                                                 size, 2)
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


def inference_pooling_s_filter(images, pool_layer_scope, kheight=2, kwidth=2):
    # channel domain pooling mapper
    split_dim = 1   # 1 represents split on spatial domain

    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                   split_dim=split_dim)
    input_image_length = len(input_image_list)
    # the pooling mapper should choose half size of the image size
    pool_s, _ = concat_eeg.pool_eeg_signal_channel(input_image_list, input_image_length/2, 1)
    _print_tensor_size(pool_s)

    # apply the normal max pooling methods with stride = 2
    pool_s = inference_pooling_n_filter(pool_s, pool_layer_scope, kheight, kwidth)

    return pool_s

def inference_pooling_s_filter(images, pool_layer_scope, kheight=2, kwidth=2):
    # channel domain pooling mapper
    split_dim = 1   # 1 represents split on spatial domain

    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                   split_dim=split_dim)
    input_image_length = len(input_image_list)
    # the pooling mapper should choose half size of the image size
    pool_s, _ = concat_eeg.pool_eeg_signal_channel(input_image_list, input_image_length/2, 1)
    _print_tensor_size(pool_s)

    # apply the normal max pooling methods with stride = 2
    pool_s = inference_pooling_n_filter(pool_s, pool_layer_scope, kheight, kwidth)

    return pool_s


def inference_pooling_t_filter(images, pool_layer_scope, kheight=2, kwidth=2):
    # channel domain pooling mapper
    split_dim = 1   # 1 represents split on spatial domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)
    input_image_length = len(input_image_list)
    # the pooling mapper should choose half size of the image size
    pool_t, _ = concat_eeg.pool_eeg_signal_channel(input_image_list, input_image_length/2, 1, is_rep=True)
    _print_tensor_size(pool_t)

    # apply the normal max pooling methods with stride = 2
    pool_t = inference_pooling_n_filter(pool_t, pool_layer_scope, kheight, kwidth)

    return pool_t


def inference_pooling_n_filter(pool_s, pool_layer_scope='pool', kheight=2, kwidth=2):
    with tf.variable_scope(pool_layer_scope, reuse=True) as scope:
        pool_s2 = tf.nn.max_pool(pool_s, ksize=[1, kheight, kwidth, 1],
                                strides=[1, kheight, kwidth, 1], padding='SAME')
        _print_tensor_size(pool_s2)

    return pool_s2


def inference_pooling_n_dropout_filter(pool_s, pool_layer_scope, kheight=2, kwidth=2):
    with tf.variable_scope(pool_layer_scope) as scope:
        pool_shape = pool_s.get_shape().as_list()

        if pool_shape[1] < 4 or pool_shape[2] < 4:
            pool_s2 = tf.nn.dropout(pool_s, 0.5)
            switches = tf.Variables(tf.ones_like(pool_s2), name='switches')
            return pool_s2

        # Recreate 1D switches for scatter update
        dim = 1

        for d in pool_shape:
            dim *= d

        ones_temp = tf.ones([(dim // kheight) // kwidth])
        temp_zeros = tf.zeros([dim])

        switches = tf.Variable(temp_zeros, name='switches')

        # clear switches
        tf.assign(switches, temp_zeros)

        [pool_s2, ind] = tf.nn.max_pool(pool_s, ksize=[1, kheight, kwidth, 1],
                                strides=[1, kheight, kwidth, 1], padding='VALID')
        _print_tensor_size(pool_s)

        switches = tf.scatter_update(switches, ind, ones_temp)

        # set switches
        switches = tf.scatter_update(switches, ind, ones_temp)

        # reshape back to batches
        switches = tf.reshape(switches, [pool_shape[0], pool_shape[1] // 2, pool_shape[2] // 2, pool_shape[3]])

        return pool_s2


def inference_pooling_L2norm_filter(images, kwidth=5):
    kheight = 2
    # channel domain pooling mapper
    split_dim = 1   # 1 represents split on spatial domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)
    input_image_length = len(input_image_list)

    if input_image_length < 16:
        images2 = tf.nn.dropout(images, 0.85)
        return images2

    # the pooling mapper should choose half size of the image size
    pool_s, _ = concat_eeg.pool_eeg_signal_channel(input_image_list, input_image_length/2, 1)
    _print_tensor_size(pool_s)

    pool_s = tf.mul(pool_s,pool_s)
    pool_s = tf.mul(float(kwidth), pool_s)

    pool_s = tf.nn.avg_pool(pool_s, ksize=[1, 1, kwidth, 1],
                            strides=[1, 1, kwidth, 1], padding='VALID')

    pool_s = tf.sqrt(pool_s)

    pool_s = tf.nn.max_pool(pool_s, ksize=[1, 2, 1, 1],
                             strides=[1, 2, 1, 1], padding='VALID')

    _print_tensor_size(pool_s)

    pool_s2 = tf.nn.dropout(pool_s, 0.85)

    return pool_s2



def inference_pooling_L2norm_choose_filter(images, kheight=2, kwidth=5):
    # channel domain pooling mapper
    split_dim = 1   # 1 represents split on spatial domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)
    input_image_length = len(input_image_list)

    # the pooling mapper should choose half size of the image size
    pool_s, _ = concat_eeg.pool_eeg_signal_channel(input_image_list, input_image_length/2, 1)
    _print_tensor_size(pool_s)

    input_shape = pool_s.get_shape()

    range_even = tf.range(0, input_shape[0], 2)
    range_odd  = tf.range(1, input_shape[0], 2)

    even_rows = tf.nn.embedding_lookup(images, range_even)
    odd_rows = tf.nn.embedding_lookup(images, range_odd)

    even_rows = tf.mul(pool_s,pool_s)
    even_rows = tf.mul(3.0, pool_s)

    even_rows = tf.nn.avg_pool(even_rows, ksize=[1, 1, 3, 1],
                            strides=[1, 1, 3, 1], padding='VALID')

    pool_s = tf.sqrt(pool_s)

    pool_s = tf.nn.max_pool(pool_s, ksize=[1, 2, 1, 1],
                             strides=[1, 2, 1, 1], padding='VALID')

    _print_tensor_size(pool_s)

    return pool_s

# endregion


# region define the fully connected layer

#def inference_fully_connected_1layer(conv_output, keep_prob):

#    # local1
#    with tf.variable_scope('local1') as scope:
#        # Move everything into depth so we can perform a single matrix multiply.
#        dim = 1

#        for d in conv_output.get_shape()[1:].as_list():
#            dim *= d

#        reshape = tf.reshape(conv_output, [FLAGS.batch_size, dim])

#        fan_in = dim
#        fan_out = 128

#        weights = _variable_with_weight_decay('weights', shape=[fan_in, fan_out],
#                                              stddev=np.sqrt(2.0/np.maximum(fan_in, fan_out)), wd=0.004 / (FLAGS.learning_rate * FLAGS.max_steps))   #0.004 works ok
#        biases = _variable_on_cpu('biases', [fan_out], tf.constant_initializer(0.001 * np.sqrt(6.0/(fan_in + fan_out))))
#        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
#        _print_tensor_size(local3)

    # dropout1
#    with tf.name_scope('dropout1'):
#        dropout1 = tf.nn.dropout(local3, keep_prob)
#        #_print_tensor_size(dropout1) # does not exist tensor shape

#    fan_in = 128
#    fan_out = 128
#    # local2
#    with tf.variable_scope('local2') as scope:
#        weights = _variable_with_weight_decay('weights', shape=[fan_in, fan_out],
#                                              stddev=np.sqrt(2.0/np.maximum(fan_in, fan_out)), wd=0.004 / (FLAGS.learning_rate * FLAGS.max_steps))   #0.004 works ok
#        biases = _variable_on_cpu('biases', [fan_out], tf.constant_initializer(0.001 * np.sqrt(6.0/(fan_in + fan_out))))
#        local4 = tf.nn.relu_layer(dropout1, weights, biases, name=scope.name)
#        _print_tensor_size(local4)

#    # dropout2
#    #with tf.name_scope('dropout2'):
#    #    dropout2 = tf.nn.dropout(local4, keep_prob)
#    #    #_print_tensor_size(dropout1) # does not exist tensor shape

#    #fan_in = 1024
#    #fan_out = 1024
#    # local3
#    #with tf.variable_scope('local3') as scope:
#    #    weights = _variable_with_weight_decay('weights', shape=[fan_in, fan_out],
#    #                                          stddev=np.sqrt(2.0/np.maximum(fan_in, fan_out)), wd=0.0 / (FLAGS.learning_rate * FLAGS.max_steps))   #0.004 works ok
#    #    biases = _variable_on_cpu('biases', [fan_out], tf.constant_initializer(0.001 * np.sqrt(6.0/(fan_in + fan_out))))
#    #    local5 = tf.nn.relu_layer(dropout2, weights, biases, name=scope.name)
#    #    _print_tensor_size(local5)

#    # local3
#    #with tf.variable_scope('local3') as scope:
#    #    weights = _variable_with_weight_decay('weights', shape=[32, 32],
#    #                                          stddev=0.04, wd=0.004)
#    #    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
#    #    local5 = tf.nn.relu_layer(dropout2, weights, biases, name=scope.name)
#    #    _print_tensor_size(local5)

#    # softmax, i.e. softmax(WX + b)
#    fan_in = 128
#    fan_out = NUM_CLASSES
#    with tf.variable_scope('softmax_linear') as scope:
#        weights = _variable_with_weight_decay('weights', [fan_in, fan_out],
#                                             stddev=np.sqrt(2.0/np.maximum(fan_in, fan_out)), wd=0.0 / (FLAGS.learning_rate * FLAGS.max_steps))
#        biases = _variable_on_cpu('biases', [fan_out], tf.constant_initializer(-0.02 * np.sqrt(6.0/(fan_in + fan_out))))
#        #logits = tf.nn.softmax(tf.matmul(local5, weights) + biases)
#        logits = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)
#        _print_tensor_size(logits)

#    return logits

# endregion



# region define the fully connected layer

def inference_fully_connected_1layer(conv_output, keep_prob):
    global batch_size
    # local1
    with tf.variable_scope('local1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in conv_output.get_shape()[1:4].as_list():
            dim *= d
        reshape = tf.reshape(conv_output, [batch_size, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 128],
                                              stddev=0.04, wd=0.004)
        #weights2 = clip_norm_fully_connected(weights, dimIn=dim, dimOut=128, perFilter=True)
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
        #weights = clip_norm_fully_connected(weights, dimIn=128, dimOut=128, perFilter=True)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(dropout1, weights, biases, name=scope.name)
        _print_tensor_size(local4)

    # dropout1
    #with tf.name_scope('dropout1'):
    #    dropout2 = tf.nn.dropout(local4, keep_prob)
        # _print_tensor_size(dropout1) # does not exist tensor shape

    # local2
    #with tf.variable_scope('local3') as scope:
    #    weights = _variable_with_weight_decay('weights', shape=[25, 16],
    #                                          stddev=0.04, wd=0.004)
    #    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    #    local5 = tf.nn.relu_layer(dropout2, weights, biases, name=scope.name)
    #    _print_tensor_size(local5)


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
        kernel = _variable_with_weight_decay('weights', shape=[5, 1, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(augment, kernel, [1, 5, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


#def inference_local_st5_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

#    #augment = inference_augment_st_filter(images, KERNEL_SIZE)
#    augment = inference_augment_s_filter(images)

#    # conv_output
#    with tf.variable_scope(conv_layer_scope) as scope:
#        images = tf.nn.l2_normalize(images, 2)
#        kernel = _variable_with_weight_decay('weights', shape=[5 , 5, in_feat, out_feat],
#                                             stddev=np.sqrt(2.0/fan_in), wd=0.0 / (FLAGS.learning_rate * FLAGS.max_steps))
#        conv = tf.nn.conv2d(augment, kernel, [1, 5, 1, 1], padding='VALID')
#        biases = _variable_with_weight_decay('biases', shape=[out_feat], stddev=np.sqrt(6.0 / fan_out), wd=0.0)
#        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
#        conv_output = tf.nn.relu(bias, name=scope.name)
#        _print_tensor_size(conv_output)
#        #conv_output = tf.nn.l2_normalize(conv_output, 2)
#
#    return conv_output

def inference_local_st5_filter(images, conv_layer_scope, in_feat=1, out_feat=4):
    #augment = inference_augment_st_filter(images, KERNEL_SIZE)
    augment = inference_augment_s_filter(images)

    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_feat, out_feat],
                                             stddev=1e-2, wd=0.0)
        conv = tf.nn.conv2d(augment, kernel, [1, 5, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output




def inference_local_st5_dropout_filter(images, conv_layer_scope, keep_prob, in_feat=1, out_feat=4):

    #augment = inference_augment_st_filter(images, KERNEL_SIZE)
    augment = inference_augment_s_filter(images)

    d1 = 5
    d2 = 2

    conv_dims = images.get_shape().as_list()

    if conv_dims[1] < 5:
        d1 = 2

    shape_tmp  = [d1, d2, in_feat, out_feat]

    dim = 1

    for d in shape_tmp:
        dim *= d

    fan_in = (dim * in_feat) // 4

    fan_out = (dim * out_feat) // 4


    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        images = tf.nn.l2_normalize(images, 2)
        kernel = _variable_with_weight_decay('weights', shape=[d1 , d2, in_feat, out_feat],
                                             stddev=np.sqrt(2.0/fan_in), wd=0.0 / (FLAGS.learning_rate * FLAGS.max_steps))

        kernel = clip_norm(kernel, kheight=5, kwidth=1, perFilter=True, out_feat=1)

        conv = tf.nn.conv2d(augment, kernel, [1, 5, 2, 1], padding='SAME')
        #biases = _variable_with_weight_decay('biases', shape=[out_feat], stddev=np.sqrt(4.0 / (fan_out)), wd=0.0)
        biases = _variable_with_weight_decay('biases', shape=[out_feat], stddev=np.sqrt(6.0 / fan_out), wd=0.0)
        #biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(np.sqrt(2.0/fan_out)))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)

        #conv_output2 = tf.nn.dropout(conv_output, keep_prob=tf.minimum(1.0 * keep_prob, 1.0))
        _print_tensor_size(conv_output)
        #conv_output = tf.nn.l2_normalize(conv_output, 2)

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

def inference_roi_s_filter_clip_norm(images, conv_layer_scope, in_feat=1, out_feat=4):

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
                                             stddev=1e-5, wd=0.0)
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
    global filterh, filterw
    # conv_output
    with tf.variable_scope(conv_layer_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[filterh, filterw, in_feat, out_feat],
                                             stddev=5e-2, wd=0.0)  #was 1e-2

        #clip_norm(kernel, kheight=filterh, kwidth=filterw, perFilter=True, in_feat=in_feat, out_feat=out_feat)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            #conv = tf.nn.dropout(conv, keep_prob=tf.minimum(keep_prob, 1.0))
        biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        #if conv_layer_scope == 'conv1':
        #    conv_output = tf.nn.dropout(conv_output, keep_prob=keep_prob)
        _print_tensor_size(conv_output)

    return conv_output

def clip_norm(kernel, kheight=filterh, kwidth=filterw, perFilter=True, in_feat=1,  out_feat = 1):
    # Clip by norm per filter
    # Calculate L2-norm
    # clip elements by ratio of clip_norm to L2-norm
    clip_norm2 = 2.0  #was 200.0
    norm2_inv = tf.rsqrt(tf.reduce_sum(kernel * kernel, [0, 1], keep_dims=True))
    # All filters  (choose this one)
    if not perFilter:
        kernel_mult = clip_norm2 * tf.reduce_min(tf.minimum(norm2_inv, 1.0 / clip_norm2), [3], keep_dims=True)
    # Per filter (or this one)
    else:
        kernel_mult = clip_norm2 * tf.minimum(norm2_inv, 1.0 / clip_norm2)

    kernel_mult_99 = tf.maximum(kernel_mult, .999999)  # was .999999 worked good on airplane/automobile


    if perFilter:
        kernel = tf.assign(kernel, kernel * tf.tile(kernel_mult_99, multiples=[kheight, kwidth, 1, 1]))  # out_feat
    else:
        kernel = tf.assign(kernel, kernel * tf.tile(kernel_mult_99, multiples=[kheight, kwidth, 1, out_feat]))  # out_feat

    return kernel



def clip_norm_fully_connected(weights, dimIn, dimOut=128, perFilter=True):
    # Clip by norm per filter
    # Calculate L2-norm
    # clip elements by ratio of clip_norm to L2-norm
    clip_norm2 = 0.2  #was 200.0
    norm2_inv = tf.rsqrt(tf.reduce_sum(weights * weights, [0], keep_dims=True))
    # All filters  (choose this one)
    if not perFilter:
        weights_mult = clip_norm2 * tf.reduce_min(tf.minimum(norm2_inv, 1.0 / clip_norm2), [1], keep_dims=True)
    # Per filter (or this one)
    else:
        weights_mult = clip_norm2 * tf.minimum(norm2_inv, 1.0 / clip_norm2)

    weights_mult_99 = tf.maximum(weights_mult, 1.0)  # was .999999 worked good on airplane/automobile
    weights_mult_99 = tf.minimum(weights_mult_99, 1.0)

    if perFilter:
        weights = tf.assign(weights, weights * tf.tile(weights_mult_99, multiples=[dimIn, 1]))  # out_feat
    else:
        weights = tf.assign(weights, weights * tf.tile(weights_mult_99, multiples=[dimIn, dimOut]))  # out_feat

    return weights

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
