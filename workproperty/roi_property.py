import gzip
import os
import sys
import urllib

import tensorflow.python.platform
import numpy
import tensorflow as tf

import warnings

WORK_DIR = '/opt/project/'
CSV_DIR = '/opt/project/data/chanlocs_csv/'

DIGIT_IMAGE_SIZE = 28

BIOSEMI_CONV = 5
BIOSEMI_POOL_ALL = 4
BIOSEMI_POOL = 2
CONV_CHAN_INFO = [256, 128, 64, 32, 16]
POOL_CHAN_INFO = [128, 64, 32, 16]


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