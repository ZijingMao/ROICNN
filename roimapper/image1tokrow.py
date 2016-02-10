from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from workproperty import roi_property


def image_1tok_kernel(vec1=np.arange(0, 28), kernelrow=5):

    # get the size of the vec1
    vec1shape = np.shape(vec1)
    vec1row = vec1shape[0]

    kernel_idx = np.zeros((vec1row, kernelrow))
    for idx in np.arange(0, vec1row):
        for idxk in np.arange(0, kernelrow):
            if idx+idxk < vec1row:
                kernel_idx[idx, idxk] = idx+idxk
            else:
                kernel_idx[idx, idxk] = vec1row-1
    kernel_idx = np.array(kernel_idx, dtype='int')

    return kernel_idx


@roi_property.deprecated
def image_1tok_row(vec1=np.arange(0, 28), kernelrow=5):

    # get the size of the vec1
    vec1shape = np.shape(vec1)
    vec1row = vec1shape[0]

    veckbyvec1 = matrix_kby1(vec1, kernelrow)

    veck = np.dot(veckbyvec1, vec1)
    veck[veck == 0] = vec1row
    veck[0] = 0

    return veck


@roi_property.deprecated
def matrix_kby1(vec1=np.arange(0, 28), kernelrow=5):
    # get the size of the vec1
    vec1shape = np.shape(vec1)
    vec1row = vec1shape[0]

    veckbyvec1 = np.zeros((kernelrow * vec1row, vec1row + kernelrow - 1))
    for idx in np.arange(0, vec1row):  # fill the veckbyvec1 with 1 in fitted slots
        for idxk in np.arange(0, kernelrow):
            fillslots = (idx*kernelrow+idxk, idx+idxk)

            veckbyvec1[fillslots] = 1  # build the mapper

    # cut the veckbyvec1 to the correct size
    veckbyvec1 = veckbyvec1[:, :vec1row]

    return veckbyvec1
