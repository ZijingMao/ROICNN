# Created by Zijing Mao at 2/10/2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import rsvp_input_data
import rsvp_quick_model
from workproperty import roi_property
import sklearn.metrics as metrics
import numpy as np

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
EEG_DATA_DIR = roi_property.FILE_DIR + \
               'rsvp_data/mat/' + EEG_DATA
EEG_TF_DIR = roi_property.FILE_DIR + \
               'rsvp_data/' + EEG_DATA
EEG_DATA_MAT = EEG_DATA_DIR + '.mat'

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 512, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.  '
                                        'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', roi_property.WORK_DIR + 'data/rsvp_train/', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                                         'for unit testing.')


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           rsvp_quick_model.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    keep_prob = tf.placeholder(tf.float32)

    return images_placeholder, labels_placeholder, keep_prob


def fill_feed_dict(data_set, drop_rate, images_pl, labels_pl, keep_prob):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        keep_prob: drop_rate
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            logits,
            images_placeholder,
            labels_placeholder,
            keep_prob,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    true_label = np.array([]).reshape(0,)   # the label information is only 1 dimension
    fake_label = np.array([]).reshape(0, 2)   # the logit information is only 2 dimensions

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   1,
                                   images_placeholder,
                                   labels_placeholder,
                                   keep_prob)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        forward_logits = sess.run(logits, feed_dict=feed_dict)  # define the logits output
        forward_labels = feed_dict[labels_placeholder]          # define the labels output
        true_label = np.concatenate((true_label, forward_labels), axis=0)
        fake_label = np.concatenate((fake_label, forward_logits), axis=0)

    # now you can calculate the auc of roc
    fpr, tpr, thresholds = metrics.roc_curve(true_label, fake_label[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)

    precision = true_count / num_examples
    # will implement AUC curve later
    # import sklearn.metrics as metrics
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, x_test[:,1], pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f  AUC @ 1: %0.04f' %
          (num_examples, true_count, precision, auc))


def run_training():
    """Train RSVP for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on RSVP.
    data_sets = rsvp_input_data.read_data_sets(EEG_DATA_MAT, FLAGS.fake_data)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder, keep_prob = placeholder_inputs(
            FLAGS.batch_size)
        # Build a Graph that computes predictions from the inference model.
        logits = rsvp_quick_model.inference(images_placeholder,
                                            FLAGS.hidden1,
                                            FLAGS.hidden2,
                                            FLAGS.hidden3,
                                            FLAGS.hidden4,
                                            keep_prob)
        # Add to the Graph the Ops for loss calculation.
        loss = rsvp_quick_model.loss(logits, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = rsvp_quick_model.training(loss, FLAGS.learning_rate)
        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = rsvp_quick_model.evaluation(logits, labels_placeholder)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)
        # And then after everything is built, start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                       0.5,
                                       images_placeholder,
                                       labels_placeholder,
                                       keep_prob)
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)
            duration = time.time() - start_time
            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 200 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        logits,
                        images_placeholder,
                        labels_placeholder,
                        keep_prob,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        logits,
                        images_placeholder,
                        labels_placeholder,
                        keep_prob,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        logits,
                        images_placeholder,
                        labels_placeholder,
                        keep_prob,
                        data_sets.test)


def main(_):
    run_training()
if __name__ == '__main__':
    tf.app.run()