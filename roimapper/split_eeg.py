# Created by Zijing Mao at 2/10/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from workproperty import roi_property
from numpy import random
import numpy as np
import tensorflow as tf

EEG_SIGNAL_SIZE = roi_property.EEG_SIGNAL_SIZE
FAKE_EEG_SIGNAL = random.randint(0, 10, size=(10, EEG_SIGNAL_SIZE, EEG_SIGNAL_SIZE, 1))
FAKE_EEG_SIGNAL = tf.constant(FAKE_EEG_SIGNAL, dtype=np.float32)


@roi_property.deprecated
def split_eeg_signal(input_eeg_signal=FAKE_EEG_SIGNAL, input_eeg_size=EEG_SIGNAL_SIZE):
    split_list = []
    # split the dimension of the axis = 1
    for split_com in tf.split(1, input_eeg_size, input_eeg_signal):
        split_list.append(split_com)
    return split_list


def split_eeg_signal_axes(input_eeg_signal=FAKE_EEG_SIGNAL, split_dim=1):
    split_list = []
    # split the dimension of the axis = 1
    eeg_signal_shape = input_eeg_signal.get_shape().as_list()
    for split_com in tf.split(split_dim, eeg_signal_shape[split_dim], input_eeg_signal):
        split_list.append(split_com)
    return split_list
