# -*- coding: utf-8 -*-
"""
Created on 3/7/16 4:47 PM
@author: Ehren Biglari
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import rsvp_quick_inference
import rsvp_quick_deconv
import scipy.io as spio
import numpy as np

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
DECONV_LASSO = 2
COMPUTER_LAST_CONV_LAYER = 3
GET_WEIGHTS = 4


#poolh = 2
#poolw = 2

poolh = 2
poolw = 2

mode = TEST

flags = tf.app.flags
FLAGS = flags.FLAGS

update_threshold = []
pool_tensor_shape = []
switches_batch = []  # Current maximum activations for feature on one image over all layers
max_activation = []  # Global maximum activations for feature on all images over all layers
max_ind = []  # Global maximum activations for feature on all images over all layers
max_image_ind = []  # Max images generate switches
cur_activation = []  # Current maximum activations for feature on one image over all layers
cur_ind = []  # Current maximum activations indicies for feature on one image over all layers
cur_image_ind = []  # Current image
max_acitvations_threshold = []  #
selection = []
max_threshold = []
pool_tensor2 = []
get_selection = []
update1 = []
update2 = []
update3 = []



def select_running_cnn(images, max_feature,
                       keep_prob, layer_num,
                       filter_num, cur_image_num, max_act_pl, max_ind_pl,
                       mode=0,
                       layer=2,
                       feat=[2, 4],
                       cnn_id=1):

    if cnn_id == DECONV_ROICNN:
        logits = deconv_roicnn(images, max_feature, keep_prob, layer_num, filter_num, cur_image_num, max_act_pl, max_ind_pl, mode, layer, feat)
    elif cnn_id == DECONV_CVCNN:
        logits = deconv_cvcnn(images, max_feature, keep_prob, layer_num, filter_num, cur_image_num, max_act_pl, max_ind_pl, mode, layer, feat)
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


def deconv_roicnn(images, max_feature, keep_prob, layer_num, filter_num,cur_image_num, max_act_pl, max_ind_pl, mode=0, layer=2, feat=[2, 4]):

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
    elif mode == DECONV_LASSO:
        results = reconstruct_input_lasso_roicnn(images, max_feature, layer_num, filter_num, max_act_pl, max_ind_pl, layer, feat)
    elif mode == COMPUTER_LAST_CONV_LAYER:
        results = compute_last_conv_layer_roicnn(images, cur_image_num, layer, feat)
    elif mode == GET_WEIGHTS:
        results = get_conv_weights(layer)

    return results



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


def test_roicnn(images, keep_prob, layer=2, feat=[2, 4]):

    for l in range(0, layer):
        if l == 0:
            conv_tensor = rsvp_quick_inference.inference_local_st5_filter(images, 'conv0', out_feat=feat[0])
        else:
            conv_tensor = rsvp_quick_inference.inference_local_st5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])

        pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=poolh, kwidth=poolw)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)

    return logits



def find_max_activation_roicnn(images, cur_image_num, layer=2, feat=[2, 4]):
    #global pool_tensor_shape, switches_batch, max_activation, max_ind, max_image_ind, cur_activation, cur_ind, cur_image_ind, max_acitvations_threshold

    #pool_tensor_shape = []
    #switches_batch -- Current maximum activations for feature on one image over all layers
    #max_activation -- Global maximum activations for feature on all images over all layers
    #max_ind -- Global maximum activations for feature on all images over all layers
    #max_image_ind -- Max images generate switches
    #cur_activation -- Current maximum activations for feature on one image over all layers
    #cur_ind -- Current maximum activations indicies for feature on one image over all layers
    #cur_image_ind -- Current image
    #max_acitvations_threshold -- a threshold for discarding maximum activations

    for l in range(0, layer):
        if l == 0:
            conv_tensor = rsvp_quick_deconv.deconv_local_st5_filter(images, 'conv0', in_feat=1, out_feat=feat[0])
        else:
            conv_tensor = rsvp_quick_deconv.deconv_local_st5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])


        with tf.variable_scope('layer' + str(l)) as scope:
            pool_tensor, _ = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=poolh, kwidth=poolw)
            pool_tensor_shape.append(pool_tensor.get_shape().as_list())
            # Initialize variables
            num_filters = pool_tensor_shape[l][3]
            max_acitvations_threshold.append(tf.Variable(tf.fill([num_filters], 10e+20 ),
                                                             name='max_acitvations_threshold'))

            max_activation.append(tf.Variable(-10e+20 * tf.ones([num_filters]), name='max_activation'))
            max_ind.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='max_ind'))
            max_image_ind.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='max_image_ind'))

            #max_activation.append(tf.Variable(tf.random_uniform(shape=[num_filters], minval=0.0001, maxval=0.0002), name='max_activation'))
            #cur_activation.append(tf.Variable(tf.zeros([num_filters]), name='cur_activation'))
            #cur_ind.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='cur_ind'))   #tf.Variable(tf.constant(-1, shape=[num_filters]))
            #cur_image_ind.append(tf.Variable(tf.fill([num_filters], tf.cast(-1, dtype=tf.int64)), name='cur_image_ind'))
            #selection.append(tf.Variable(tf.fill([num_filters], tf.constant(False)), name='selection'))
            #max_threshold.append(tf.Variable(tf.zeros([1,pool_tensor_shape[l][1], pool_tensor_shape[l][2], pool_tensor_shape[l][3]]), name='max_threshold'))
            #max_threshold.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='max_ind'))

            cur_image_ind.append(tf.fill([pool_tensor_shape[l][3]], cur_image_num))

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

            updated_max_ind = tf.select(selection[l], cur_ind[l], max_ind[l])
            update1.append(tf.assign(max_ind[l], updated_max_ind))
            updated_max_image_ind = tf.select(selection[l], cur_image_ind[l], max_image_ind[l])
            update2.append(tf.assign(max_image_ind[l], updated_max_image_ind))
            updated_max_activations = tf.select(selection[l], cur_activation[l], max_activation[l])
            update3.append(tf.assign(max_activation[l], updated_max_activations))


    returnTensors = []
    returnTensors.extend(cur_activation)
    returnTensors.extend(selection)
    returnTensors.extend(update1)
    returnTensors.extend(update2)
    returnTensors.extend(update3)
    returnTensors.extend(max_acitvations_threshold)
    returnTensors.extend(pool_tensor)
    returnTensors.extend(max_ind)
    returnTensors.extend(max_image_ind)
    returnTensors.extend(max_activation)

    #returnTensors.extend(cur_image_ind)
    #returnTensors.extend(cur_activation)
    #returnTensors.extend(cur_ind)
    #returnTensors.extend(max_ind)
    #returnTensors.extend(max_threshold)
    #returnTensors.extend([pool_tensor])
    #returnTensors.extend([pool_tensor2])
    #returnTensors.extend(max_acitvations_threshold)
    #returnTensors.extend([pool_tensor3])

    update_max_threshold(layer=2)

    clear_variables(layer=2)

    return returnTensors


def reconstruct_input_lasso_roicnn(images, max_feature, layer_num, filter_num, max_act_pl, max_ind_pl, layer, feat=[2, 4]):
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

        pool_tensor, switches_tmp = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=poolh, kwidth=poolw)
        pool_tensor_shape.append(pool_tensor.get_shape().as_list())
        switches.append(switches_tmp)

    deconv_tensor = max_feature

    for l in range(layer_num, -1, -1):
        unpool_tensor = rsvp_quick_deconv.deconv_unpooling_n_filter(deconv_tensor , switches[l], 'pool' + str(l), kheight=poolh, kwidth=poolw)

        if l == 0:
            deconv_tensor = rsvp_quick_deconv.deconv_local_st5_unfilter(unpool_tensor, conv_tensor_input_shape[l], 'conv0')
        else:
            deconv_tensor = rsvp_quick_deconv.deconv_local_st5_unfilter \
                (unpool_tensor, conv_tensor_input_shape[l], 'conv' + str(l))

    returnTensors = []
    #returnTensors.extend([max_act_val]   )
    #returnTensors.extend([max_ind_val])
    returnTensors.extend([deconv_tensor])
    returnTensors.extend([pool_tensor])


    returnTensors.extend(switches)

    return returnTensors


def compute_last_conv_layer_roicnn(images, cur_image_num, layer=2, feat=[2, 4]):
    for l in range(0, layer):
        if l == 0:
            conv_tensor = rsvp_quick_deconv.deconv_local_st5_filter(images, 'conv0', in_feat=1, out_feat=feat[0])
        else:
            conv_tensor = rsvp_quick_deconv.deconv_local_st5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])


        with tf.variable_scope('layer' + str(l)) as scope:
            pool_tensor, _ = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=poolh, kwidth=poolw)
            pool_tensor_shape.append(pool_tensor.get_shape().as_list())


    returnTensors = []
    returnTensors.extend([pool_tensor])

    return returnTensors


def get_conv_weights(layer=2):
    w = []
    b = []
    lw1 = []
    lb1 = []
    lw2 = []
    lb2 = []
    mw = []
    mb = []

    for l in range(0, layer):
        wtmp, btmp = rsvp_quick_deconv.deconv_get_weights('conv' + str(l))
        w.append(wtmp)
        b.append(btmp)

    for l in range(2, layer+2):
        # local layers
        wtmp, btmp = rsvp_quick_deconv.deconv_get_weights('local' + str(l-1))
        w.append(wtmp)
        b.append(btmp)

    # Softmax Linear
    wtmp, btmp = rsvp_quick_deconv.deconv_get_weights('softmax_linear')
    w.append(wtmp)
    b.append(btmp)

    returnTensors = []
    returnTensors.extend(w)
    returnTensors.extend(b)
    b.append(btmp)

    return returnTensors

def save_conv_weights(fname, allWeights, layer, layer_fc):
    for l in xrange(0, layer + layer_fc + 1):
        w = np.array([allWeights[l]])
        b = np.array([allWeights[l + layer + layer_fc + 1]])

        spio.savemat('/home/e/LASSOdeconv/lassoresults/' + fname + str(l) + '.mat', dict(weights=w, biases=b))


def load_conv_weights(layername, layer=2, layer_fc=2):
    w = []
    b = []
    w2 = []
    b2 = []
    for l in range(0, layer+3):
        # f = scipy.io.loadmat(train_data)
        # f = scipy.io.loadmat('/home/e/deconvresults/preprocess/RSVP_X2_S01_RAW_CH64_preproc.mat')
        f = spio.loadmat('/home/e/LASSOdeconv/lassoresults/' + layername + str(l) + '.mat')
        layerweights_np = np.squeeze(f["weights"][:], (0,))
        layerbiases_np = np.squeeze(f["biases"][:])
        print("size of loaded weights: " + str(np.shape(layerweights_np)))
        print("size of loaded biases: " + str(np.shape(layerbiases_np)))
        weights_const = tf.constant(layerweights_np)
        biases_const = tf.constant(layerbiases_np)
        w.append(weights_const)
        b.append(biases_const)

        if (l < 2):
            layer_name = 'conv' + str(l)
        elif l < (layer + layer_fc):
            layer_name = 'local' + str(l - (layer-1))
        else:
            layer_name = 'softmax_linear'

        with tf.variable_scope(layer_name, reuse=True) as scope:
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            w.append(tf.assign(kernel, weights_const))
            b.append(tf.assign(biases, biases_const))


    returnTensors = []
    returnTensors.extend(w)
    returnTensors.extend(b)

    return returnTensors




def deconv_cvcnn(images, max_feature, keep_prob, layer_num, filter_num,cur_image_num, max_act_pl, max_ind_pl, mode=0, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_roicnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
        print('Make sure you have defined the feature map size for each layer.')
        return

    # local st
    switches_shape = []
    switches_batch = []



    if mode == TEST:
        results = test_cvcnn(images, keep_prob, layer, feat)
    elif mode == FIND_MAX_ACTIVATION_GET_SWITCHES:
        results = find_max_activation_cvcnn(images, cur_image_num, layer, feat)
    elif mode == DECONV_LASSO:
        results = reconstruct_input_lasso_cvcnn(images, max_feature, keep_prob, layer_num, filter_num, max_act_pl, max_ind_pl, layer, feat)
    elif mode == COMPUTER_LAST_CONV_LAYER:
        results = compute_last_conv_layer_cvcnn(images, cur_image_num, keep_prob, layer, feat)
    elif mode == GET_WEIGHTS:
        results = get_conv_weights(layer)

    return results



def inference_cvcnn(images, keep_prob, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_cvcnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
        print('Make sure you have defined the feature map size for each layer.')
        return

    # local st
    conv_tensor = rsvp_quick_inference.inference_5x5_filter(images, 'conv0', keep_prob, in_feat=1, out_feat=feat[0])
    pool_tensor, switches_tmp = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool0', kheight=poolh, kwidth=poolw)
    for l in range(1, layer):
        conv_tensor = rsvp_quick_inference.inference_5x5_filter\
            (pool_tensor, 'conv'+str(l), keep_prob, in_feat=feat[l-1], out_feat=feat[l])
        pool_tensor, switches_tmp = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool'+str(l), kheight=poolh, kwidth=poolw)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def test_cvcnn(images, keep_prob, layer=2, feat=[2, 4]):

    for l in range(0, layer):
        if l == 0:
            conv_tensor = rsvp_quick_inference.inference_5x5_filter(images, 'conv0', keep_prob, out_feat=feat[0])
        else:
            conv_tensor = rsvp_quick_inference.inference_5x5_filter \
                (pool_tensor, 'conv' + str(l), keep_prob, in_feat=feat[l - 1], out_feat=feat[l])

        pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=poolh, kwidth=poolw)  # was 1 x 4

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)

    return logits



def find_max_activation_cvcnn(images, cur_image_num, layer=2, feat=[2, 4]):
    #global pool_tensor_shape, switches_batch, max_activation, max_ind, max_image_ind, cur_activation, cur_ind, cur_image_ind, max_acitvations_threshold

    #pool_tensor_shape = []
    #switches_batch -- Current maximum activations for feature on one image over all layers
    #max_activation -- Global maximum activations for feature on all images over all layers
    #max_ind -- Global maximum activations for feature on all images over all layers
    #max_image_ind -- Max images generate switches
    #cur_activation -- Current maximum activations for feature on one image over all layers
    #cur_ind -- Current maximum activations indicies for feature on one image over all layers
    #cur_image_ind -- Current image
    #max_acitvations_threshold -- a threshold for discarding maximum activations

    for l in range(0, layer):
        if l == 0:
            conv_tensor = rsvp_quick_deconv.deconv_5x5_filter(images, 'conv0', in_feat=1, out_feat=feat[0])
        else:
            conv_tensor = rsvp_quick_deconv.deconv_5x5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])


        with tf.variable_scope('layer' + str(l)) as scope:
            pool_tensor, _ = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=poolh, kwidth=poolw)
            pool_tensor_shape.append(pool_tensor.get_shape().as_list())
            # Initialize variables
            num_filters = pool_tensor_shape[l][3]
            max_acitvations_threshold.append(tf.Variable(tf.fill([num_filters], 10e+20 ),
                                                             name='max_acitvations_threshold'))

            max_activation.append(tf.Variable(-10e+20 * tf.ones([num_filters]), name='max_activation'))
            max_ind.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='max_ind'))
            max_image_ind.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='max_image_ind'))

            #max_activation.append(tf.Variable(tf.random_uniform(shape=[num_filters], minval=0.0001, maxval=0.0002), name='max_activation'))
            #cur_activation.append(tf.Variable(tf.zeros([num_filters]), name='cur_activation'))
            #cur_ind.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='cur_ind'))   #tf.Variable(tf.constant(-1, shape=[num_filters]))
            #cur_image_ind.append(tf.Variable(tf.fill([num_filters], tf.cast(-1, dtype=tf.int64)), name='cur_image_ind'))
            #selection.append(tf.Variable(tf.fill([num_filters], tf.constant(False)), name='selection'))
            #max_threshold.append(tf.Variable(tf.zeros([1,pool_tensor_shape[l][1], pool_tensor_shape[l][2], pool_tensor_shape[l][3]]), name='max_threshold'))
            #max_threshold.append(tf.Variable(tf.fill([num_filters], tf.constant(-1, dtype=tf.int64)), name='max_ind'))

            cur_image_ind.append(tf.fill([pool_tensor_shape[l][3]], cur_image_num))

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

            updated_max_ind = tf.select(selection[l], cur_ind[l], max_ind[l])
            update1.append(tf.assign(max_ind[l], updated_max_ind))
            updated_max_image_ind = tf.select(selection[l], cur_image_ind[l], max_image_ind[l])
            update2.append(tf.assign(max_image_ind[l], updated_max_image_ind))
            updated_max_activations = tf.select(selection[l], cur_activation[l], max_activation[l])
            update3.append(tf.assign(max_activation[l], updated_max_activations))


    returnTensors = []
    #returnTensors.extend(cur_activation)
    #returnTensors.extend(selection)
    #returnTensors.extend(update1)
    #returnTensors.extend(update2)
    #returnTensors.extend(update3)
    #returnTensors.extend(max_acitvations_threshold)
    #returnTensors.extend(pool_tensor])
    #returnTensors.extend(max_ind)
    #returnTensors.extend(max_image_ind)
    #returnTensors.extend(max_activation)

    #returnTensors.extend(cur_image_ind)
    #returnTensors.extend(cur_activation)
    #returnTensors.extend(cur_ind)
    #returnTensors.extend(max_ind)
    #returnTensors.extend(max_threshold)
    #returnTensors.extend([pool_tensor])
    #returnTensors.extend([pool_tensor2])
    #returnTensors.extend(max_acitvations_threshold)
    #returnTensors.extend([pool_tensor3])

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


def reconstruct_input_lasso_cvcnn(images, max_feature, keep_prob, layer_num, filter_num, max_act_pl, max_ind_pl, layer, feat=[2, 4]):
    switches = []
    pool_tensor_shape = []
    conv_tensor_input_shape = []
    pool_tensors = []
    deconv_tensors = []
    unpool_tensors = []
    unpool_resize_tensors  = []
    conv_tensors = []

    for l in range(0, layer_num + 1):
        if l == 0:
            conv_tensor = rsvp_quick_deconv.deconv_5x5_filter(images, 'conv0', in_feat=1, out_feat=feat[0])
            conv_tensor_input_shape.append(images.get_shape().as_list())
        else:
            conv_tensor = rsvp_quick_deconv.deconv_5x5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])
            conv_tensor_input_shape.append(pool_tensor.get_shape().as_list())

        conv_tensors.append(conv_tensor)
        pool_tensor, switches_tmp = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=poolh, kwidth=poolw)
        pool_tensor_shape.append(pool_tensor.get_shape().as_list())
        pool_tensors.append(pool_tensor)
        switches.append(switches_tmp)

    # deconv_tensor = max_feature

    if (layer_num == 1):
        logits, layer1, layer2 = rsvp_quick_deconv.deconv_fully_connected_1layer(pool_tensor, keep_prob)

    deconv_tensor = max_feature


    for l in range(layer_num, -1, -1):
        unpool_tensor, unpool_resize_tensor = rsvp_quick_deconv.deconv_unpooling_n_filter(deconv_tensor , switches[l], 'pool' + str(l), kheight=poolh, kwidth=poolw)

        unpool_resize_tensors.append(unpool_resize_tensor)
        unpool_tensors.append(unpool_tensor)

        if l == 0:
            deconv_tensor = rsvp_quick_deconv.deconv_5x5_unfilter(unpool_tensor, conv_tensor_input_shape[l], 'conv0')
        else:
            deconv_tensor = rsvp_quick_deconv.deconv_5x5_unfilter \
                (unpool_tensor, conv_tensor_input_shape[l], 'conv' + str(l))

        deconv_tensors.append(deconv_tensor)

    returnTensors = []
    #returnTensors.extend([max_act_val]   )
    #returnTensors.extend([max_ind_val])
    returnTensors.extend(deconv_tensors)
    returnTensors.extend(pool_tensors)
    returnTensors.extend(switches)
    returnTensors.extend(unpool_tensors)
    returnTensors.extend(unpool_resize_tensors)
    returnTensors.extend(conv_tensors)

    if (layer_num == 1):
        returnTensors.extend([logits, layer1, layer2])

    return returnTensors


def compute_last_conv_layer_cvcnn(images, cur_image_num, keep_prob, layer=2, feat=[2, 4]):
    for l in range(0, layer):
        if l == 0:
            conv_tensor = rsvp_quick_deconv.deconv_5x5_filter(images, 'conv0', in_feat=1, out_feat=feat[0])
        else:
            conv_tensor = rsvp_quick_deconv.deconv_5x5_filter \
                (pool_tensor, 'conv' + str(l), in_feat=feat[l - 1], out_feat=feat[l])


        with tf.variable_scope('layer' + str(l)) as scope:
            pool_tensor, _ = rsvp_quick_deconv.deconv_pooling_n_filter(conv_tensor, 'pool' + str(l), kheight=poolh, kwidth=poolw)
            pool_tensor_shape.append(pool_tensor.get_shape().as_list())

    logits, layer1, layer2 = rsvp_quick_deconv.deconv_fully_connected_1layer(pool_tensor, keep_prob)

    returnTensors = []
    returnTensors.extend([pool_tensor])
    returnTensors.extend([layer2])
    returnTensors.extend([logits])

    return returnTensors