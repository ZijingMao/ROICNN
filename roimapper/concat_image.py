# Created by Zijing Mao at 2/8/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from workproperty import roi_property
import image1tokrow
import numpy as np
import tensorflow as tf

DIGIT_IMAGE_SIZE = roi_property.DIGIT_IMAGE_SIZE


def concat_digit_image(input_image_list, vec_idx=np.arange(0, 28), kernelrow=5, concat_dim1=1, concat_dim2=3):
    if not isinstance(input_image_list, list):
        print("input image is not a list.")

    if len(input_image_list) > 0:
        input_shape = input_image_list[0].get_shape().as_list()
    else:
        print("input image not exist.")
        return

    # get the size of the vec1
    vec1shape = np.shape(vec_idx)
    vec1row = vec1shape[0]  # 28 here

    # input_shape = [10, 1, 28, 1]
    input_shape[concat_dim1] *= kernelrow
    input_shape[concat_dim2] *= vec1row
    # input_shape = [10, 5, 28, 28]

    image_kernel_idx = image1tokrow.image_1tok_kernel(vec_idx, kernelrow)
    curr_kernel_tensor = []
    for kernel_idx in image_kernel_idx:  # go for every kernel => 28 kernels
        # for each kernel => 5 index
        # concat on the first dimension => concat_dim1 = 1
        curr_kernel_tensor.append(tf.concat(concat_dim1, [input_image_list[idx] for idx in kernel_idx]))

    kernel_tensor = tf.concat(concat_dim2, curr_kernel_tensor)

    return kernel_tensor, input_shape
