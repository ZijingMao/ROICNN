# -*- coding: utf-8 -*-
"""
Created on 3/7/16 4:47 PM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rsvp_quick_inference


def _print_tensor_size(given_tensor):
    # print the shape of tensor
    print("="*78)
    print("Tensor Name: " + given_tensor.name)
    print(given_tensor.get_shape().as_list())


def inference_roicnn(images, keep_prob):

    _print_tensor_size(images)
    assert isinstance(keep_prob, object)

    # local st
    conv1 = rsvp_quick_inference.inference_local_st5_filter(images, 'conv1', out_feat=8)
    pool1 = rsvp_quick_inference.inference_pooling_s_filter(conv1)
    conv1 = rsvp_quick_inference.inference_local_st5_filter(pool1, 'conv2', in_feat=8, out_feat=8)
    pool1 = rsvp_quick_inference.inference_pooling_s_filter(conv1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool1, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_tcnn(images, keep_prob):

    _print_tensor_size(images)
    assert isinstance(keep_prob, object)

    # local st
    conv1 = rsvp_quick_inference.inference_local_st5_filter(images, 'conv1', out_feat=8)
    pool1 = rsvp_quick_inference.inference_pooling_s_filter(conv1)
    conv1 = rsvp_quick_inference.inference_local_st5_filter(pool1, 'conv2', in_feat=8, out_feat=8)
    pool1 = rsvp_quick_inference.inference_pooling_s_filter(conv1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool1, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_cvcnn(images, keep_prob):

    _print_tensor_size(images)
    assert isinstance(keep_prob, object)

    # local st
    conv1 = rsvp_quick_inference.inference_local_st5_filter(images, 'conv1', out_feat=8)
    pool1 = rsvp_quick_inference.inference_pooling_s_filter(conv1)
    conv1 = rsvp_quick_inference.inference_local_st5_filter(pool1, 'conv2', in_feat=8, out_feat=8)
    pool1 = rsvp_quick_inference.inference_pooling_s_filter(conv1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool1, keep_prob)

    assert isinstance(logits, object)
    return logits


def inference_stcnn(images, keep_prob):

    _print_tensor_size(images)
    assert isinstance(keep_prob, object)

    # local st
    conv1 = rsvp_quick_inference.inference_local_st5_filter(images, 'conv1', out_feat=8)
    pool1 = rsvp_quick_inference.inference_pooling_s_filter(conv1)
    conv1 = rsvp_quick_inference.inference_local_st5_filter(pool1, 'conv2', in_feat=8, out_feat=8)
    pool1 = rsvp_quick_inference.inference_pooling_s_filter(conv1)

    logits = rsvp_quick_inference.inference_fully_connected_1layer(pool1, keep_prob)

    assert isinstance(logits, object)
    return logits
