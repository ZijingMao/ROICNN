"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets