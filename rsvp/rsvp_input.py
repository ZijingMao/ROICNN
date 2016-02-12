# Created by Zijing Mao at 2/10/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from workproperty import roi_property


def read_rsvp(filename_queue):
    """Reads and parses examples from rsvp data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of channel in the result (64/256)
        width: number of time in the result (64/256)
        depth: number of features in the result (1)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..64.
        float64: a [height, width, depth] float64 Tensor with the image data
    """

    class RSVPRecord(object):
        pass
    result = RSVPRecord()

    # Dimensions of the images in the RSVP dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    result.height = roi_property.EEG_SIGNAL_SIZE
    result.width = roi_property.EEG_SIGNAL_SIZE
    result.depth = 1
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the RSVP format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.TFRecordReader()
    result.key, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      dense_keys=['image_raw', 'label'],
      # Defaults are not specified since both keys are required.
      dense_types=[tf.string, tf.int64])

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.float64)
    image.set_shape([image_bytes])
    image = tf.reshape(image, [result.height, result.width, result.depth])

    label = tf.cast(features['label'], tf.int32)

    result.image = image
    result.label = label

    return result
