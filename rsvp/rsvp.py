# Created by Zijing Mao at 2/10/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import re
import sys
import tarfile
import urllib

import tensorflow.python.platform
import tensorflow as tf

import cifar10_input
from tensorflow.python.platform import gfile
import roi_property

FLAGS = tf.app.flags.FLAGS

WORK_DIRECTORY = roi_property.WORK_DIR + 'data/rsvp_data'

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of eeg epochs to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', WORK_DIRECTORY,
                           """Path to the rsvp data directory.""")


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the rsvp data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'zijing.mao@cheetah.cbi.utsa.edu'


def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        print('Refer to http://www.eegstudy.org/ for download.')

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
