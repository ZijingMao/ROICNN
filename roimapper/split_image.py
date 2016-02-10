# Created by Zijing Mao at 2/8/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from workproperty import roi_property
from numpy import random
import numpy as np
import tensorflow as tf

DIGIT_IMAGE_SIZE = roi_property.DIGIT_IMAGE_SIZE
FAKE_DIGIT_IMAGE = random.randint(0, 10, size=(10, DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1))
FAKE_DIGIT_IMAGE = tf.constant(FAKE_DIGIT_IMAGE, dtype=np.float32)


def split_digit_image(input_digit_image=FAKE_DIGIT_IMAGE):
    split_list = []
    # split the dimension of the axis = 1
    for split_com in tf.split(1, DIGIT_IMAGE_SIZE, input_digit_image):
        split_list.append(split_com)
    return split_list


def split_digit_image_axes(input_digit_image=FAKE_DIGIT_IMAGE, split_dim=1):
    split_list = []
    # split the dimension of the axis = 1
    for split_com in tf.split(split_dim, DIGIT_IMAGE_SIZE, input_digit_image):
        split_list.append(split_com)
    return split_list

