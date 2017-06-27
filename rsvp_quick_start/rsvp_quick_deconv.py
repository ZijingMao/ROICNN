# -*- coding: utf-8 -*-
"""
Created on 2/17/16 3:29 PM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from workproperty import roi_property
from roimapper import concat_eeg
from roimapper import split_eeg
import numpy as np

# The RSVP dataset has 2 classes, representing the digits 0 through 1.
NUM_CLASSES = roi_property.BINARY_LABEL  # replace with multiple labels
IMAGE_SIZE = roi_property.EEG_SIGNAL_SIZE
KERNEL_SIZE = roi_property.BIOSEMI_CONV

flags = tf.app.flags
FLAGS = flags.FLAGS

# region define the variable methods

batch_size = 1


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
    augment = deconv_augment_s_filter(images)

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
    augment = deconv_augment_s_filter(images)

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


def deconv_augment_s_filter(images):
    # recommend to use
    # augment
    split_dim = 1   # 1 represents split on spatial domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)
    input_image_length = len(input_image_list)
    augment, _ = concat_eeg.conv_eeg_signal_channel(input_image_list, input_image_length, 1)
    _print_tensor_size(augment)

    return augment




def deconv_pooling_n_filter(pool_s, pool_layer_scope, kheight=2, kwidth=2):
    with tf.variable_scope(pool_layer_scope, reuse=True) as scope:
        pool_shape = pool_s.get_shape().as_list()
        #if pool_shape[1] < 4 or pool_shape[2] < 4:
        #    pool_s2 = tf.nn.dropout(pool_s, 0.5)
        #    switches = tf.ones_like(pool_s2)
        #    return pool_s2
        #Recreate 1D switches for scatter update
        dim = 1

        for d in pool_shape:
            dim *= d

        [pool_s2, ind] = tf.nn.max_pool_with_argmax(pool_s, ksize=[1, kheight, kwidth, 1],
                                strides=[1, kheight, kwidth, 1], padding='SAME')

        _print_tensor_size(pool_s2)

        #ones_temp = tf.ones_like([(dim // kheight) // kwidth])
        ones_temp = tf.ones_like(ind, dtype=tf.float32)
        #temp_zeros =

        switches = tf.Variable(tf.zeros([dim]), name='switches')

        switches = tf.assign(switches, tf.zeros([dim]))

        # set switches
        switches_out2 = tf.scatter_update(switches, ind, ones_temp)

        #reshape back to batches
        switches_out2 = tf.reshape(switches_out2, pool_shape)

    return pool_s2, switches_out2


def deconv_unpooling_n_filter(unpool_s, switches, unpool_layer_scope, kheight=2, kwidth=2):
    with tf.variable_scope(unpool_layer_scope, reuse=True) as scope:
        pool_shape = unpool_s.get_shape().as_list()
        switches_shape = switches.get_shape().as_list()

        #Recreate 1D switches for scatter update
        dim = 1

        for d in pool_shape:
            dim *= d

        ones_temp = tf.ones([(dim // kheight) // kwidth])
        temp_zeros = tf.zeros([dim])

        unpool_s_resize = tf.image.resize_nearest_neighbor(unpool_s, [switches_shape[1], switches_shape[2]], align_corners=False)

        unpool_s2 = tf.multiply(unpool_s_resize, switches)

        _print_tensor_size(unpool_s2)

    return unpool_s2, unpool_s_resize


# endregion




def deconv_fully_connected_1layer(conv_output, keep_prob):
    global batch_size
    # local1
    with tf.variable_scope('local1', reuse=True ) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in conv_output.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(conv_output, [batch_size, dim])

        weights = tf.get_variable('weights')
        biases = tf.get_variable('biases')
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        _print_tensor_size(local3)

    # dropout1
    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(local3, keep_prob)
        # _print_tensor_size(dropout1) # does not exist tensor shape

    # local2
    with tf.variable_scope('local2', reuse=True) as scope:
        weights = tf.get_variable('weights')
        biases = tf.get_variable('biases')
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
    with tf.variable_scope('softmax_linear', reuse=True) as scope:
        weights = tf.get_variable('weights')
        biases = tf.get_variable('biases')
        logits = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)
        _print_tensor_size(logits)

    return logits, local3, local4





# region define 1-layer modules here

def deconv_local_st5_filter(images, conv_layer_scope, in_feat, out_feat):

    #augment = inference_augment_st_filter(images, KERNEL_SIZE)
    augment = deconv_augment_s_filter(images)

    # conv_output
    with tf.variable_scope(conv_layer_scope, reuse=True) as scope:
        #kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_feat, out_feat],
        #                                     stddev=1e-2, wd=0.0)
        #biases = _variable_on_cpu('biases', [out_feat], tf.constant_initializer(0.0))
        kernel = tf.get_variable('weights')
        biases = tf.get_variable('biases')
        conv = tf.nn.conv2d(augment, kernel, [1, 5, 1, 1], padding='SAME')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def deconv_get_weights(layer_scope):

    # conv_output
    with tf.variable_scope(layer_scope, reuse=True) as scope:
        weights = tf.get_variable('weights')
        biases = tf.get_variable('biases')

    return weights, biases



def deconv_local_st5_unfilter(images, output_shape, conv_layer_scope):

    #augment = inference_augment_st_filter(images, KERNEL_SIZE)
    augment = deconv_augment_s_filter(images)
    images = augment

    # conv_output
    with tf.variable_scope(conv_layer_scope, reuse=True) as scope:
        images_shape = images.get_shape().as_list()
        inv_map = tf.range(0, images_shape[0]*images_shape[1]*images_shape[2]*output_shape[3])
        inv_map = tf.reshape(inv_map, shape=output_shape)
        inv_map = deconv_augment_s_filter(inv_map)
        inv_map_1d = tf.reshape(inv_map, shape=[-1])

        kernel = tf.get_variable('weights')
        biases = tf.get_variable('biases')

        relu_output = tf.nn.relu(images, name=scope.name)
        bias = tf.nn.bias_add(relu_output, -biases)

        #deconv = tf.reshape(tf.nn.bias_add(relu_output, -biases), augment.get_shape().as_list())
        #deconv_output = tf.nn.conv2d(relu_output*25, tf.transpose(kernel, perm=[0, 1, 3, 2]), [1, 5, 1, 1], padding='SAME')
        deconv_output = tf.nn.conv2d_transpose(bias, kernel, output_shape=[output_shape[0], output_shape[1]*5, output_shape[2], output_shape[3]], strides=[1, 5, 1, 1], padding='SAME')

        deconv_output_1d = tf.reshape(deconv_output, [-1])
        tmp_output = tf.Variable(tf.zeros(shape=[output_shape[0]*output_shape[1]*output_shape[2]*output_shape[3]], dtype=tf.float32), name='tmp_output')
        tmp_output = tf.assign(tmp_output, tf.zeros(shape=[output_shape[0]*output_shape[1]*output_shape[2]*output_shape[3]], dtype=tf.float32))
        tmp_output2 = tf.scatter_add(tmp_output, inv_map_1d, deconv_output_1d)
        tmp_output3 = tmp_output2 * (1.0/5.0)
        tmp_output4 = tf.reshape(tmp_output3, shape=output_shape)

        _print_tensor_size(tmp_output4)

    return tmp_output4


def deconv_5x5_filter(images, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    with tf.variable_scope(conv_layer_scope, reuse=True) as scope:
        kernel = tf.get_variable('weights')
        biases = tf.get_variable('biases')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv_output = tf.nn.relu(bias, name=scope.name)
        _print_tensor_size(conv_output)

    return conv_output


def deconv_5x5_unfilter(images, output_shape, conv_layer_scope, in_feat=1, out_feat=4):

    # conv_output
    with tf.variable_scope(conv_layer_scope, reuse=True) as scope:
        kernel = tf.get_variable('weights')
        biases = tf.get_variable('biases')
        relu_output = tf.nn.relu(images, name=scope.name)
        deconv = tf.reshape(tf.nn.bias_add(relu_output, -biases), images.get_shape().as_list())
        #deconv3 = tf.nn.relu(images, name=scope.name)
        #deconv2 = tf.reshape(tf.nn.bias_add(deconv3, -biases), images.get_shape().as_list())
        #deconv1 = tf.nn.relu(deconv2, name=scope.name)
        deconv_output = tf.nn.conv2d_transpose(relu_output, kernel, output_shape=[output_shape[0], output_shape[1], output_shape[2], output_shape[3]], strides=[1, 1, 1, 1], padding='SAME')

        #deconv_output = tf.nn.conv2d(deconv, kernel, tf.transpose(kernel, perm=[0, 1, 3, 2]), [1, 1, 1, 1], padding='SAME')
        _print_tensor_size(deconv_output)

    return deconv_output

# endregion
