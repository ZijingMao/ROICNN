# Created by Zijing Mao at 2/10/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from workproperty import roi_property
from workproperty import roi_mapper_idx
import numpy as np
import tensorflow as tf
import image1tokrow
import biosemi_chan_mapper

EEG_SIGNAL_SIZE = roi_property.EEG_SIGNAL_SIZE
BIOSEMI_POOL = roi_property.BIOSEMI_POOL


def conv_eeg_signal_channel(
        input_eeg_list,
        channel_idx,
        concat_dim1=1):
    '''
    The function will use a bunch of image tensor and concat them with the multiplication of kernel size and from the
    channel dimension as concat_dim1=1. The arrangement should be every eeg height (channel) contains 5 kernel indexes
    Args:
        input_eeg_list: input image tensor list that obtained from split_image.py
        channel_idx: the channel index based on different channel information
        concat_dim1: concat dimension 1, the channel dimension

    Returns: the concated kernel tensor and the tensor input shape for verification

    '''
    if not isinstance(input_eeg_list, list):
        print("input eeg split signal is not a list of tensor.")

    if len(input_eeg_list) > 0:
        input_shape = input_eeg_list[0].get_shape().as_list()
    else:
        print("input eeg split not exist.")
        return

    # get the size of the vec1
    # conv_idx = biosemi_chan_mapper.conv_mapper(channel_idx)
    conv_idx = roi_mapper_idx.roi_mapper_idx_select(channel_idx, 'conv')
    conv_idx = biosemi_chan_mapper.replace_mapper_idx(conv_idx)
    conv_idx_shape = conv_idx.shape     # (256, 5)

    # input_shape = [10, 1, 256, 1]
    input_shape[concat_dim1] *= conv_idx_shape[0]
    input_shape[concat_dim1] *= conv_idx_shape[1]
    # input_shape = [10, 5*256, 256, 1]

    curr_kernel_tensor = []
    for kernel_idx in conv_idx:  # go for every kernel => 256 kernels
        # for each kernel => 5 index
        # concat on the first dimension => concat_dim1 = 1
        curr_kernel_tensor.append(tf.concat(concat_dim1, [input_eeg_list[idx] for idx in kernel_idx]))

    kernel_tensor = tf.concat(concat_dim1, curr_kernel_tensor)

    return kernel_tensor, input_shape


def conv_eeg_signal_time(
        input_eeg_list,
        vec_idx=np.arange(0, EEG_SIGNAL_SIZE),
        kernelrow=roi_property.BIOSEMI_CONV,
        concat_dim1=2, is_rep=False):
    '''
    The function will use a bunch of image tensor and concat them with the multiplication of kernel size and from the
    time dimension as concat_dim1=2. The arrangement should be every eeg width (time) contains 5 kernel indexes.
    Args:
        input_eeg_list: input image tensor list that obtained from split_image.py
        vec_idx: the length of the vector used for mapping
        kernelrow: the kernel size
        concat_dim1: concat dimension 2, the time dimension
        is_rep: only used in testing local roi on spatial domain

    Returns: the concated kernel tensor and the tensor input shape for verification

    '''
    if not isinstance(input_eeg_list, list):
        print("input eeg split signal is not a list of tensor.")

    if len(input_eeg_list) > 0:
        input_shape = input_eeg_list[0].get_shape().as_list()
    else:
        print("input eeg split not exist.")
        return

    # get the size of the vec1
    vec1shape = np.shape(vec_idx)
    vec1row = vec1shape[0]  # 256/128/64/32/16 here

    # input_shape = [10, 1, 256, 1]
    input_shape[concat_dim1] *= kernelrow
    input_shape[concat_dim1] *= vec1row
    # input_shape = [10, 5*256, 256, 1]
    if is_rep:
        vec_idx = np.reshape(vec_idx, (vec1row, 1))
        image_kernel_idx = np.repeat(vec_idx, kernelrow, axis=1)
    else:
        image_kernel_idx = image1tokrow.image_1tok_kernel(vec_idx, kernelrow)

    curr_kernel_tensor = []
    for kernel_idx in image_kernel_idx:  # go for every kernel => 256 kernels
        # for each kernel => 5 index
        # concat on the first dimension => concat_dim1 = 1
        curr_kernel_tensor.append(tf.concat(concat_dim1, [input_eeg_list[idx] for idx in kernel_idx]))

    kernel_tensor = tf.concat(concat_dim1, curr_kernel_tensor)

    return kernel_tensor, input_shape


def pool_eeg_signal_channel(
        input_eeg_list,
        channel_idx,
        concat_dim1=1,
        is_rep=True):
    '''
    The function will use a bunch of image tensor and concat them with the multiplication of pooling size from the
    channel dimension as concat_dim1=1. The arrangement should be every eeg height (channel) contains 5 kernel indexes
    Args:
        input_eeg_list: input image tensor list that obtained from split_image.py
        channel_idx: the channel index based on different channel information
        concat_dim1: concat dimension 1, the channel dimension
        is_rep: True indicate using direct mapper

    Returns: the concated kernel tensor and the tensor input shape for verification

    '''
    if not isinstance(input_eeg_list, list):
        print("input eeg split signal is not a list of tensor.")

    if len(input_eeg_list) > 0:
        input_shape = input_eeg_list[0].get_shape().as_list()
    else:
        print("input eeg split not exist.")
        return

    # get the size of the vec1
    # conv_idx = biosemi_chan_mapper.pool_mapper(channel_idx)

    if is_rep:  # this will only be used for pooling of resize channels
        # vec_idx = np.arange(0, EEG_SIGNAL_SIZE)
        # vec_idx = np.reshape(vec_idx, (EEG_SIGNAL_SIZE, 1))
        vec_idx = roi_mapper_idx.roi_mapper_idx_select(channel_idx)
        vec_idx = biosemi_chan_mapper.replace_mapper_idx(vec_idx)
        vec_idx = np.array([row[0] for row in vec_idx])
        vec_idx = np.reshape(vec_idx, (vec_idx.__len__(), 1))
        conv_idx = np.repeat(vec_idx, BIOSEMI_POOL, axis=1)
    else:
        conv_idx = roi_mapper_idx.roi_mapper_idx_select(channel_idx)
        conv_idx = biosemi_chan_mapper.replace_mapper_idx(conv_idx)
    conv_idx_shape = conv_idx.shape     # (128, 2)

    # input_shape = [10, 1, 256, 1]
    input_shape[concat_dim1] *= conv_idx_shape[0]
    input_shape[concat_dim1] *= conv_idx_shape[1]

    # input_shape = [10, 2*128, 256, 1]

    curr_kernel_tensor = []
    for kernel_idx in conv_idx:  # go for every kernel => 256 kernels
        # for each kernel => 5 index
        # concat on the first dimension => concat_dim1 = 1
        if kernel_idx[0] == 64:
            print('')
        tempzero = [input_eeg_list[idx] for idx in kernel_idx]
        tempone = tf.concat(concat_dim1, tempzero)
        curr_kernel_tensor.append(tempone)

    kernel_tensor = tf.concat(concat_dim1, curr_kernel_tensor)

    return kernel_tensor, input_shape


def pool_eeg_signal_time():
    # will be implemented on demands
    pass
