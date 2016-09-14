# Created by Zijing Mao at 2/12/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow.python.platform
import tensorflow as tf
from workproperty import roi_property
import rsvp_quick_inference
from autorun import autorun_infer

# The RSVP dataset has 2 classes, representing the digits 0 through 1.
NUM_CLASSES = roi_property.BINARY_LABEL
IMAGE_SIZE = roi_property.EEG_SIGNAL_SIZE

flags = tf.app.flags
FLAGS = flags.FLAGS


def _print_tensor_size(given_tensor):
    # print the shape of tensor
    print("="*78)
    print("Tensor Name: " + given_tensor.name)
    print(given_tensor.get_shape().as_list())


def inference(images, keep_prob):
    """Build the RSVP model up to where it may be used for inference.
    Args:
      images (object): Images placeholder, from inputs().
      keep_prob: Dropout placeholder
    Returns:
      softmax_linear: Output tensor with the computed logits.
    """

    # TODO check the inference data structure, the size has to be a square
    _print_tensor_size(images)
    assert isinstance(keep_prob, object)

    # local st

    # global st
    # conv1 = rsvp_quick_inference.inference_global_st_filter(images, 'conv1', out_feat=4)
    # pool1 = rsvp_quick_inference.inference_pooling_n_filter(conv1, kheight=1)
    # conv1 = rsvp_quick_inference.inference_temporal_filter(pool1, 'conv2', in_feat=4, out_feat=4)
    # pool1 = rsvp_quick_inference.inference_pooling_n_filter(conv1, kheight=1)

    # local cv
    # conv1 = rsvp_quick_inference.inference_5x5_filter(images, 'conv1', out_feat=128)
    # pool1 = rsvp_quick_inference.inference_pooling_n_filter(conv1, kheight=2)
    # conv1 = rsvp_quick_inference.inference_5x5_filter(pool1, 'conv2', in_feat=128, out_feat=4)
    # pool1 = rsvp_quick_inference.inference_pooling_n_filter(conv1, kheight=2)
    # conv1 = rsvp_quick_inference.inference_1x1_filter(pool1, 'conv3', in_feat=4, out_feat=4)
    # pool1 = rsvp_quick_inference.inference_pooling_n_filter(conv1, kheight=2)

    # logits = rsvp_quick_inference.inference_fully_connected_1layer(pool1, keep_prob)

    logits = autorun_infer.inference_roi_ts_cnn(images, keep_prob, layer=2, feat=[2, 64])

    assert isinstance(logits, object)
    return logits


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    # Convert from sparse integer labels in the range [0, NUM_CLASSES)
    # to 1-hot dense float vectors (that is we will have batch_size vectors,
    # each with NUM_CLASSES values, all of which are 0.0 except there will
    # be a 1.0 in the entry corresponding to the label).
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                           onehot_labels,
                                                            name='xentropy')


    #cross_entropy = tf.nn.l2_loss(logits - onehot_labels)

    # Calculate the average cross entropy loss across the batch.
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(
        learning_rate,                # Base learning rate.
        global_step,  # Current index into the dataset.
        1000,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(lr, 0.9)  # was .35  #.7 works okay, gets to .79 by step 400 at 0.03 learning rate with ROI
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))

