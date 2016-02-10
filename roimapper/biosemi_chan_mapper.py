# Created by Zijing Mao at 2/9/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from workproperty import roi_property
import numpy as np
import csv


def conv_mapper(chan_num):
    '''

    Args:
        chan_num: the channel number will take 256, 128, 64, 32, and 16

    Returns:
        it will return a 2-D array of index for channels
    '''
    reader = csv.reader(open(roi_property.CSV_DIR + 'Idx' + str(chan_num) + 'Chan5Kernel.csv', 'rb'), delimiter=',')
    reader_list = list(reader)
    conv_idx = np.array(reader_list).astype('int')
    return conv_idx


def pool_mapper(chan_num, pool_size=roi_property.BIOSEMI_POOL):
    '''

    Args:
        chan_num: the channel number will take 128, 64, 32, and 16
        pool_size: the kernel size for the pooling padding

    Returns:
        it will return a 2-D array of index in channels for the pooling data
    '''
    reader = csv.reader(open(roi_property.CSV_DIR + 'Idx' + str(chan_num) + 'ChanPool.csv', 'rb'), delimiter=',')
    reader_list = list(reader)
    pool_idx = np.array(reader_list).astype('int')
    pool_idx = pool_idx[:, :pool_size]
    return pool_idx


def replace_mapper_idx(mapper_idx, strategy='central'):
    '''

    Args:
        mapper_idx: the mapper used for indexing spliced tensors
        strategy: the strategy support 'null': do nothing, 'central': replace by the central element, others might be
        implemented in the future

    Returns:

    '''
    mapper_shape = mapper_idx.shape
    if strategy == 'null':
        pass
    elif strategy == 'central':
        for idx in np.arange(0, mapper_shape[0]):
            zero_idx = mapper_idx[idx, :] == 0
            # replace the 0 value with the first element of the row
            mapper_idx[idx, zero_idx] = mapper_idx[idx, 0]
    else:
        pass

    # the true index is started from 0 not 1
    mapper_idx -= 1

    return mapper_idx
