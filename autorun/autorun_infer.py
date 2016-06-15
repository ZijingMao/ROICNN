# -*- coding: utf-8 -*-
"""
Created on 3/7/16 4:47 PM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rsvp_quick_inference

# region define inference name
INFERENCE_ROICNN        = 0
INFERENCE_CVCNN         = 1
INFERENCE_LOCAL_T_CNN   = 2
INFERENCE_LOCAL_S_CNN   = 3
INFERENCE_GLOBAL_T_CNN  = 4
INFERENCE_GLOBAL_S_CNN  = 5
INFERENCE_DNN_CNN       = 6
INFERENCE_STCNN         = 7
INFERENCE_TSCNN         = 8
INFERENCE_ROI_S_CNN     = 9
INFERENCE_ROI_TS_CNN    = 10
# endregion


def select_running_cnn(images,
                       keep_prob,
                       layer=2,
                       feat=[2,4],
                       cnn_id=1):
    if cnn_id == INFERENCE_ROICNN:
        logits = inference_roicnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_CVCNN:
        logits = inference_cvcnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_LOCAL_T_CNN:
        logits = inference_local_t_cnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_LOCAL_S_CNN:
        logits = inference_local_s_cnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_GLOBAL_T_CNN:
        logits = inference_global_t_cnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_GLOBAL_S_CNN:
        logits = inference_global_s_cnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_DNN_CNN:
        logits = inference_dnn_cnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_STCNN:
        logits = inference_stcnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_TSCNN:
        logits = inference_tscnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_ROI_S_CNN:
        logits = inference_roi_s_cnn(images, keep_prob, layer, feat)
    elif cnn_id == INFERENCE_ROI_TS_CNN:
        logits = inference_roi_ts_cnn(images, keep_prob, layer, feat)
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


def inference_roicnn(images, keep_prob, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_roicnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
        print('Make sure you have defined the feature map size for each layer.')
        return

    # local st
    conv_tensor = rsvp_quick_inference.inference_local_st5_filter(images, 'conv0', out_feat=feat[0])
    # conv_tensor = rsvp_quick_inference.inference_pooling_direct_map_filter(conv_tensor, kwidth=2)
    conv_tensor = rsvp_quick_inference.inference_pooling_direct_map_filter(conv_tensor, kheight=1, kwidth=4)
    for l in range(1, layer):
        # conv_tensor = rsvp_quick_inference.inference_time_wise_filter(conv_tensor, 'conv1', in_feat=feat[0], out_feat=feat[1])
        # the pooling should have the height padding to 1 because no channel anymore
        # conv_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kheight=1, kwidth=2)
        conv_tensor = rsvp_quick_inference.inference_local_st5_filter\
            (conv_tensor, 'conv'+str(l), in_feat=feat[l-1], out_feat=feat[l])
        conv_tensor = rsvp_quick_inference.inference_pooling_direct_map_filter(conv_tensor, kwidth=2)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(conv_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_roi_s_cnn(images, keep_prob, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_roi_s_cnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
        print('Make sure you have defined the feature map size for each layer.')
        return

    # local st
    conv_tensor = rsvp_quick_inference.inference_roi_s_filter(images, 'conv0', out_feat=feat[0])
    # pool_tensor = rsvp_quick_inference.inference_pooling_s_filter(conv_tensor)
    # for l in range(1, layer):
    #     conv_tensor = rsvp_quick_inference.inference_roi_s_filter\
    #         (pool_tensor, 'conv'+str(l), in_feat=feat[l-1], out_feat=feat[l])
    #     pool_tensor = rsvp_quick_inference.inference_pooling_s_filter(conv_tensor)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(conv_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_roi_ts_cnn(images, keep_prob, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_roi_ts_cnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
        print('Make sure you have defined the feature map size for each layer.')
        return

    # local st
    conv_tensor = rsvp_quick_inference.inference_roi_global_ts_filter(images, 'conv0', out_feat=feat[0])
    pool_tensor = rsvp_quick_inference.inference_pooling_s_filter(conv_tensor, kwidth=1)
    for l in range(1, layer):
        conv_tensor = rsvp_quick_inference.inference_roi_s_filter\
            (pool_tensor, 'conv'+str(l), in_feat=feat[l-1], out_feat=feat[l])
        pool_tensor = rsvp_quick_inference.inference_pooling_s_filter(conv_tensor, kwidth=1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_cvcnn(images, keep_prob, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_cvcnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
        print('Make sure you have defined the feature map size for each layer.')
        return

    # local st
    conv_tensor = rsvp_quick_inference.inference_5x5_filter(images, 'conv0', out_feat=feat[0])
    pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor)
    for l in range(1, layer):
        conv_tensor = rsvp_quick_inference.inference_5x5_filter\
            (pool_tensor, 'conv'+str(l), in_feat=feat[l-1], out_feat=feat[l])
        pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_local_t_cnn(images, keep_prob, layer=1, feat=[2]):

    _print_tensor_size(images, 'inference_local_t_cnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
        print('Make sure you have defined the feature map size for each layer.')
        return

    # local t
    # here use the 1*5 filter which go across channels
    conv_tensor = rsvp_quick_inference.inference_temporal_filter(images, 'conv0', out_feat=feat[0])
    # the pooling should have the width padding to 1 because we only consider channel correlation
    pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kwidth=1)
    for l in range(1, layer):
        # here use the 1*5 filter which go across channels
        conv_tensor = rsvp_quick_inference.inference_temporal_filter\
            (pool_tensor, 'conv'+str(l), in_feat=feat[l-1], out_feat=feat[l])
        # the pooling should have the width padding to 1 because we only consider channel correlation
        pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kwidth=1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_local_s_cnn(images, keep_prob, layer=1, feat=[2]):

    _print_tensor_size(images, 'inference_local_s_cnn')
    assert isinstance(keep_prob, object)

    if not layer == len(feat):
            print('Make sure you have defined the feature map size for each layer.')
            return

    # local t
    # here use the 1*5 filter which go across channels
    conv_tensor = rsvp_quick_inference.inference_spatial_filter(images, 'conv0', out_feat=feat[0])
    # the pooling should have the width padding to 1 because we only consider channel correlation
    pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kheight=1)
    for l in range(1, layer):
        # here use the 1*5 filter which go across channels
        conv_tensor = rsvp_quick_inference.inference_spatial_filter\
            (pool_tensor, 'conv'+str(l), in_feat=feat[l-1], out_feat=feat[l])
        # the pooling should have the width padding to 1 because we only consider channel correlation
        pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kheight=1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_global_t_cnn(images, keep_prob, layer=1, feat=[4]):

    _print_tensor_size(images, 'inference_global_t_cnn')
    assert isinstance(keep_prob, object)

    # global t
    # here use the channel wise filter which go across channels
    conv_tensor = rsvp_quick_inference.inference_channel_wise_filter(images, 'conv1', out_feat=feat[0])
    # the pooling should have the width padding to 1 because no width anymore
    pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kwidth=1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_global_s_cnn(images, keep_prob, layer=1, feat=[4]):

    _print_tensor_size(images, 'inference_global_s_cnn')
    assert isinstance(keep_prob, object)

    # global s
    # here use the spatial filter which go across time
    conv_tensor = rsvp_quick_inference.inference_time_wise_filter(images, 'conv1', out_feat=feat[0])
    # the pooling should have the height padding to 1 because no channel anymore
    pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kheight=1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_dnn_cnn(images, keep_prob, layer=1, feat=[4]):

    _print_tensor_size(images, 'inference_dnn_cnn')
    assert isinstance(keep_prob, object)

    # apply the global filter on both temporal & spatial domain
    logits = rsvp_quick_inference.inference_fully_connected_1layer(images, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_stcnn(images, keep_prob, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_stcnn')
    assert isinstance(keep_prob, object)

    # global spatial local temporal
    conv_tensor = rsvp_quick_inference.inference_global_st_filter(images, 'conv0', out_feat=feat[0])
    pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kheight=1)

    for l in range(1, layer):
        # here use the 1*5 filter which go across channels
        conv_tensor = rsvp_quick_inference.inference_temporal_filter\
            (pool_tensor, 'conv'+str(l), in_feat=feat[l-1], out_feat=feat[l])
        # the pooling should have the width padding to 1 because we only consider channel correlation
        pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kheight=1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_tscnn(images, keep_prob, layer=2, feat=[2, 4]):

    _print_tensor_size(images, 'inference_tscnn')
    assert isinstance(keep_prob, object)

    # global temporal local temporal
    conv_tensor = rsvp_quick_inference.inference_global_ts_filter(images, 'conv0', out_feat=feat[0])
    pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kwidth=1)

    for l in range(1, layer):
        # here use the 1*5 filter which go across channels
        conv_tensor = rsvp_quick_inference.inference_spatial_filter\
            (pool_tensor, 'conv'+str(l), in_feat=feat[l-1], out_feat=feat[l])
        # the pooling should have the width padding to 1 because we only consider channel correlation
        pool_tensor = rsvp_quick_inference.inference_pooling_n_filter(conv_tensor, kwidth=1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool_tensor, keep_prob)

    assert isinstance(logits, object)
    return logits
