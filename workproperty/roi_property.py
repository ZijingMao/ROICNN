import gzip
import os
import sys
import urllib

import math
import tensorflow.python.platform
import numpy
import tensorflow as tf

import warnings

WORK_DIR = '/home/zijing/jingxia/ROICNN/'
# WORK_DIR = '/home/eeglab/PycharmProjects/ROICNN/'
CSV_DIR = WORK_DIR + 'data/chanlocs_csv/'
FILE_DIR = WORK_DIR + 'data/'
SAVE_DIR = WORK_DIR + 'result/'

DIGIT_IMAGE_SIZE = 64

EEG_SIGNAL_SIZE = 28
BIOSEMI_CONV = 5
BIOSEMI_POOL_ALL = 4
BIOSEMI_POOL = 2
CONV_CHAN_INFO = [256, 128, 64, 32, 16]
POOL_CHAN_INFO = [128, 64, 32, 16]

LAYER_LIST = range(1, int(math.log(EEG_SIGNAL_SIZE, 2))-2)
# FEAT_LIST = [2**j for j in range(3, 7)]
FEAT_LIST = [2**j for j in range(2, 6)]
MAX_RAND_SEARCH = 10

BATCH_SIZE = 128
# BATCH_SIZE = 64
# BATCH_SIZE =1 FOR DECONVOLUTION
# BATCH_SIZE = 1
SMALL_TRAIN_SIZE = 1000
SMALL_VALID_SIZE = 100
SMALL_TEST_SIZE = 100
MEDIUM_TRAIN_SIZE = 10000
MEDIUM_VALID_SIZE = 1000
MEDIUM_TEST_SIZE = 1000
LARGE_TRAIN_SIZE = 100000
LARGE_VALID_SIZE = 10000
LARGE_TEST_SIZE = 10000
HUGE_TRAIN_SIZE = 1000000
HUGE_VALID_SIZE = 100000
HUGE_TEST_SIZE = 100000
# BINARY_LABEL = 108
BINARY_LABEL = 157
MULTI_LABEL = 157

# EXP_TYPE_STR = ['DRIVE']
EXP_TYPE_STR = ['FOUR']
# EXP_NAME_STR = ['XB']
EXP_NAME_STR = ['COM']
DAT_TYPE_STR = ['NORMFREQ', 'FREQ', 'NORM', 'RAWFREQ', 'RAW', 'SUB', 'AVG']
# DAT_TYPE_STR = ['FREQNORM', 'FREQ', 'NORM', 'CS_RAW', 'FREQRAW', 'SUB', 'AVG', 'CS_SPD']
# SUB_STR = ['S'+str(sub).zfill(2) for sub in range(1, 1000)]
SUB_STR = ['S'+str(sub).zfill(2) for sub in range(157,158)]
CHAN_STR = 'CH'+str(EEG_SIGNAL_SIZE)


def deprecated(func):
    '''
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    '''
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func