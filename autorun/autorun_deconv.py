# -*- coding: utf-8 -*-
"""
Created on 5/25/16 3:48 PM
@author: Ehren Biglari
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import rsvp_quick_inference
import rsvp_quick_deconv

# region define inference name
DECONV_ROICNN        = 0
DECONV_CVCNN         = 1
#DECONV_LOCAL_T_CNN   = 2
#DECONV_LOCAL_S_CNN   = 3
#DECONV_GLOBAL_T_CNN  = 4
#DECONV_GLOBAL_S_CNN  = 5
#DECONV_DNN_CNN       = 6
#DECONV_STCNN         = 7
#DECONV_TSCNN         = 8
#DECONV_ROI_S_CNN     = 9
#DECONV_ROI_TS_CNN    = 10
# endregion

# deconv modes
TEST = 0
FIND_MAX_ACTIVATION_GET_SWITCHES = 1
RECONSTRUCT_INPUT = 2


mode = TEST

flags = tf.app.flags
FLAGS = flags.FLAGS

update_threshold = []  # Threshold for update
pool_tensor_shape = [] # Pool tensor shape
switches_batch = []    # Current maximum activations for feature on one image over all layers
max_activation = []    # Global maximum activations for feature on all images over all layers
max_ind = []           # Global maximum activations for feature on all images over all layers
max_image_ind = []     # Max images generate switches
cur_activation = []    # Current maximum activations for feature on one image over all layers
cur_ind = []           # Current maximum activations indicies for feature on one image over all layers
cur_image_ind = []     # Current image
max_acitvations_threshold = []  # Maximum activation threshold
selection = []         # Selection of top features for current iteration
max_threshold = []     # Maximum threshold for current iteration
pool_tensor2 = []      # Pool tensor for current iteration
get_selection = []     # Tensor for assigning selection
update1 = []
update2 = []
update3 = []


def select_running_cnn(images,
                       keep_prob, layer_num,
                       filter_num, cur_image_num, max_act_pl, max_ind_pl,
                       mode=0,
                       layer=2,
                       feat=[2, 4],
                       cnn_id=1):

    if cnn_id == DECONV_ROICNN:
        logits = deconv_roicnn(images, keep_prob, layer_num, filter_num, cur_image_num, max_act_pl, max_ind_pl, mode, layer, feat)
    else:
        logits = None
        print("unrecognized cnn model, make sure you have the correct inference")

    return logits

def _print_tensor_size(given_tensor, inference_name=""):
    # print the shape of tensor
    print("="*78)
    print("Model: " + "\t"*3 + inference_name)
    print("Tensor Name: \t" + given_tensor.name)
    print(given_tensor.get_shape().as_list())


def deconv_roicnn(images, keep_prob, layer_num, filter_num,cur_image_num, max_act_pl, max_ind_pl, mode=0, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_roicnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
        print('Make sure you have defined the feature map size for each layer.')
        return

    # local st
    switches_shape = []
    switches_batch = []



    if mode == TEST:
        results = test_roicnn(images, keep_prob, layer, feat)
    elif mode == FIND_MAX_ACTIVATION_GET_SWITCHES:
        results = find_max_activation_roicnn(images, cur_image_num, layer, feat)
    elif mode == RECONSTRUCT_INPUT:
        results = reconstruct_input_roicnn(images, layer_num, filter_num, max_act_pl, max_ind_pl, layer, feat)

    return results


def test_roicnn(images, keep_prob, layer=2, feat=[2, 4]):

    for l in range(0, layer):
        if l == 0:
            conv_tensor = rsvp_quick_inference.inference_local_st5_filter(images, 'conv0', out_feat=feat[0])
        else:
            conv_tensor = rsvp_quick_inference.inference_local_st5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])

        pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=1, kwidth=4)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)

    return logits


def fake_array_of_tensors(layer, name):
    fake_list = []
    for l in range(0, layer):
        with tf.variable_scope('layer' + str(l), reuse=True) as scope:
            tf.Variable(0, name='max_ind')

    return fake_list

def fake_logits():
    # Create a fake empty logits
    with tf.variable_scope('softmax_linear', reuse=True) as scope:
        logits = tf.Variable(0, name=scope.name)

    return logits


def find_max_activation_roicnn(images, cur_image_num, layer=2, feat=[2, 4]):
    for l in range(0, layer):
        if l == 0:
            conv_tensor = rsvp_quick_deconv.deconv_local_st5_filter(images, 'conv0', in_feat=1, out_feat=feat[0])
        else:
            conv_tensor = rsvp_quick_deconv.deconv_local_st5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])


        with tf.variable_scope('layer' + str(l)) as scope:
            pool_tensor, _ = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=2, kwidth=2)
            pool_tensor_shape.append(pool_tensor.get_shape().as_list())
            # Initialize variables
            num_filters = pool_tensor_shape[l][3]
            max_acitvations_threshold.append(tf.Variable(tf.fill([num_filters], 10e+20 ),
                                                             name='max_acitvations_threshold'))

            max_activation.append(tf.Variable(-10e+20 * tf.ones([num_filters]), name='max_activation'))
            max_ind.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='max_ind'))
            max_image_ind.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='max_image_ind'))


            cur_image_ind.append(tf.fill([pool_tensor_shape[l][3]], cur_image_num))

            # Get maximum activations
            max_acitvations_threshold_tmp1 = tf.expand_dims(max_acitvations_threshold[l], 0)
            max_acitvations_threshold_tmp2 = tf.expand_dims(max_acitvations_threshold_tmp1, 0)
            max_acitvations_threshold_tmp3 = tf.expand_dims(max_acitvations_threshold_tmp2, 0)

            max_threshold.append(tf.tile(max_acitvations_threshold_tmp3, [1,pool_tensor_shape[l][1], pool_tensor_shape[l][2], 1]))
            pool_tensor2 = tf.select(max_threshold[l] >= pool_tensor, pool_tensor, -10e+20 * tf.ones_like(pool_tensor))
            cur_activation.append(tf.reduce_max(pool_tensor2, [0, 1, 2]))
            pool_tensor2 = tf.transpose(pool_tensor2, [3, 1, 2, 0])
            pool_tensor3 = tf.reshape(pool_tensor2, [pool_tensor_shape[l][3], pool_tensor_shape[l][1] * pool_tensor_shape[l][2]])
            cur_ind.append(tf.argmax(pool_tensor3, 1))


            selection.append(tf.logical_and(cur_activation[l] > max_activation[l],
                                       max_acitvations_threshold[l] > cur_activation[l]))

            # Update maximum activations
            updated_max_ind = tf.select(selection[l], cur_ind[l], max_ind[l])
            update1.append(tf.assign(max_ind[l], updated_max_ind))
            updated_max_image_ind = tf.select(selection[l], cur_image_ind[l], max_image_ind[l])
            update2.append(tf.assign(max_image_ind[l], updated_max_image_ind))
            updated_max_activations = tf.select(selection[l], cur_activation[l], max_activation[l])
            update3.append(tf.assign(max_activation[l], updated_max_activations))

    returnTensors = []
    # Return data regarding maximums
    returnTensors.extend(cur_activation)
    returnTensors.extend(selection)
    returnTensors.extend(update1)
    returnTensors.extend(update2)
    returnTensors.extend(update3)
    returnTensors.extend(max_acitvations_threshold)
    returnTensors.extend(max_ind)
    returnTensors.extend(max_image_ind)
    returnTensors.extend(max_activation)

    update_max_threshold(layer=2)

    clear_variables(layer=2)

    return returnTensors


clear_vars = []
update_threshold = []

clear_max_activation = []
clear_max_ind = []
clear_max_image_ind = []

def clear_variables(layer=2):

    for l in range(0, layer):
        with tf.variable_scope('layer' + str(l)) as scope:
            clear_max_activation.append(tf.assign(max_activation[l], -10e+20 * tf.ones_like(max_activation[l])))
            clear_max_ind.append(tf.assign(max_ind[l], -1 * tf.ones_like(max_ind[l])))
            clear_max_image_ind.append(tf.assign(max_image_ind[l], -1 * tf.ones_like(max_image_ind[l])))

    clear_vars.extend(clear_max_image_ind)
    clear_vars.extend(clear_max_ind)
    clear_vars.extend(clear_max_activation)


    return

def update_max_threshold(layer=2):
    for l in range(0, layer):
        with tf.variable_scope('layer' + str(l)) as scope:
            update_threshold.append(tf.assign(max_acitvations_threshold[l], max_activation[l]))

    return

def get_update_threshold():
    return update_threshold


def get_clear_variables():
    return clear_vars


def reconstruct_input_roicnn(images, layer_num, filter_num, max_act_pl, max_ind_pl, layer, feat=[2, 4]):
    switches = []
    pool_tensor_shape = []
    conv_tensor_input_shape = []

    for l in range(0, layer_num + 1):
        if l == 0:
            conv_tensor = rsvp_quick_deconv.deconv_local_st5_filter(images, 'conv0', in_feat=1, out_feat=feat[0])
            conv_tensor_input_shape.append(images.get_shape().as_list())
        else:
            conv_tensor = rsvp_quick_deconv.deconv_local_st5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])
            conv_tensor_input_shape.append(pool_tensor.get_shape().as_list())

        pool_tensor, switches_tmp = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=2, kwidth=2)
        pool_tensor_shape.append(pool_tensor.get_shape().as_list())
        switches.append(switches_tmp)

        if l == layer_num:
            with tf.variable_scope('toplayer' + str(l)) as scope:
		# Set top layer activations based on maximum activations
                max_act_feat = tf.Variable(tf.zeros([pool_tensor_shape[l][3]*pool_tensor_shape[l][1]*pool_tensor_shape[l][2]]), name='max_act_feat')
                max_act_feat = tf.assign(max_act_feat, tf.zeros([pool_tensor_shape[l][3]*pool_tensor_shape[l][1]*pool_tensor_shape[l][2]]))
                max_features_tmp = tf.scatter_update(max_act_feat, max_ind_pl + filter_num * pool_tensor_shape[l][1] * pool_tensor_shape[l][2], max_act_pl)
                max_features_tmp2 = tf.reshape(max_features_tmp, [pool_tensor_shape[l][3], pool_tensor_shape[l][1],pool_tensor_shape[l][2]])
                max_features_tmp3 = tf.transpose(max_features_tmp2, [1, 2, 0])
                max_feature = tf.expand_dims(max_features_tmp3, 0)


    deconv_tensor = max_feature

    # Deconvolution network
    for l in range(layer_num, -1, -1):
        unpool_tensor = rsvp_quick_deconv.deconv_unpooling_n_filter(deconv_tensor , switches[l], 'pool' + str(l), kheight=2, kwidth=2)

        if l == 0:
            deconv_tensor = rsvp_quick_deconv.deconv_local_st5_unfilter(unpool_tensor, conv_tensor_input_shape[l], 'conv0')
        else:
            deconv_tensor = rsvp_quick_deconv.deconv_local_st5_unfilter \
                (unpool_tensor, conv_tensor_input_shape[l], 'conv' + str(l))

    returnTensors = []
    returnTensors.extend([max_act_feat])
    returnTensors.extend([max_feature])
    returnTensors.extend([deconv_tensor])

    returnTensors.extend(switches)

    return returnTensors


