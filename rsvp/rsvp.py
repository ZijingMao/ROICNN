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

import rsvp_input
from tensorflow.python.platform import gfile
import roi_property

FLAGS = tf.app.flags.FLAGS

EXP_TYPE_STR = roi_property.EXP_TYPE_STR[0]
EXP_NAME_STR = roi_property.EXP_NAME_STR[0]
DAT_TYPE_STR = roi_property.DAT_TYPE_STR[0]
SUB_STR = roi_property.SUB_STR[0]
CHAN_STR = roi_property.CHAN_STR

EEG_DATA = EXP_TYPE_STR + '_' + \
           EXP_NAME_STR + '_' + \
           SUB_STR + '_' + \
           DAT_TYPE_STR + '_' + \
           CHAN_STR

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', roi_property.BATCH_SIZE,
                            """Number of eeg epochs to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', roi_property.FILE_DIR + 'rsvp_data/',
                           """Path to the rsvp data directory.""")


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
EEG_SIGNAL_SIZE = roi_property.EEG_SIGNAL_SIZE

# Global constants describing the rsvp data set.
NUM_CLASSES = roi_property.BINARY_LABEL
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = roi_property.SMALL_TRAIN_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = roi_property.SMALL_TEST_SIZE

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'zijing.mao@cheetah.cbi.utsa.edu:~/Data/'

TRAIN_FILE = EEG_DATA+'.train.tfrecords'
VALIDATION_FILE = EEG_DATA+'.validation.tfrecords'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _generate_image_and_label_batch(image, label, min_queue_examples):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 1] of type.float64.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'FLAGS.batch_size' images + labels from the example queue.
    num_preprocess_threads = 12
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [FLAGS.batch_size])


def inputs(train):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    filename = os.path.join(FLAGS.train_dir,
                            TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        filename_q = tf.train.string_input_producer([filename])

        # Even when reading in multiple threads, share the filename
        # queue.
        result = rsvp_input.read_rsvp(filename_q)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in multi-threads to avoid being a bottleneck.

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        print ('Filling queue with %d RSVP data before starting to train. '
               'This will take a few minutes.' % min_queue_examples)
        images, sparse_labels = \
            _generate_image_and_label_batch(result.image, result.label, min_queue_examples)

        return images, sparse_labels


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Raises:
      ValueError: if no data_dir

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(FLAGS.data_dir, EEG_DATA+'.train.tfrecords')]
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = rsvp_input.read_rsvp(filename_queue)
    reshaped_image = tf.cast(read_input.image, tf.float64)

    height = EEG_SIGNAL_SIZE
    width = EEG_SIGNAL_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.image.random_crop(reshaped_image, [height, width])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples)


def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        print('Refer to http://www.eegstudy.org/ if not downloaded.')

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
