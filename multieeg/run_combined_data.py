# -*- coding: utf-8 -*-
"""
Created on 3/26/16 10:16 AM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

import autorun_infer
import autorun_util
import rsvp_input_data
import rsvp_quick_cnn_model
from workproperty import roi_property
import sklearn.metrics as metrics
import numpy as np

import tensorflow as tf

# EXP_TYPE_STR = roi_property.EXP_TYPE_STR[0]
# EXP_NAME_STR = roi_property.EXP_NAME_STR[0]
# DAT_TYPE_STR = roi_property.DAT_TYPE_STR[0]
# SUB_STR = roi_property.SUB_STR[0]
# CHAN_STR = roi_property.CHAN_STR

# Basic model parameters as external flags.
# TODO try to change learning rate in the rsvp folder
EEG_TF_DIR = roi_property.FILE_DIR + \
               'rsvp_data/rand_search'
learning_rate = 0.006
choose_cnn_type = 1
batch_size = 128
max_step = 5000    # to guarantee 64 epochs # should be training sample_size
check_step = max_step/50

layer_list = roi_property.LAYER_LIST
feat_list = roi_property.FEAT_LIST
max_rand_search = roi_property.MAX_RAND_SEARCH

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', max_step, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', batch_size, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', roi_property.WORK_DIR + 'data/rsvp_train/', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                                         'for unit testing.')


def placeholder_inputs(batch_size, feat_size=1):
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
                                                           rsvp_quick_cnn_model.IMAGE_SIZE,
                                                           rsvp_quick_cnn_model.IMAGE_SIZE,
                                                           feat_size))
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
      drop_rate: the dropout rate that will be used in training, for
                 testing it should just set to 1
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
      keep_prob: the probability that will be used for dropout
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
            data_set,
            csv_writer_acc=None,
            csv_writer_auc=None):
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
    # write the csv file if exists
    if csv_writer_acc is not None:
        csv_writer_acc.write('%0.06f' % precision)
        csv_writer_acc.write('\n')
    if csv_writer_auc is not None:
        csv_writer_auc.write('%0.06f' % auc)
        csv_writer_auc.write('\n')


def run_training(hyper_param, model, name_idx, sub_idx):
    '''
    Train RSVP for a number of steps.
    Args:
        hyper_param: three elements, layer & feat & model
        model:
        name_idx:
        sub_idx:

    Returns:

    '''
    # initialize the summary to write
    csv_writer_acc, csv_writer_auc = autorun_util.csv_writer\
        (model, hyper_param['feat'], name_idx=name_idx, sub_idx=sub_idx)
    # Get the sets of images and labels for training, validation, and
    # test on RSVP.
    eeg_data = autorun_util.str_name(name_idx, sub_idx)
    eeg_data_dir = roi_property.FILE_DIR + \
                   'rsvp_data/mat_x2/' + eeg_data
    eeg_data_mat = eeg_data_dir + '.mat'
    data_sets = rsvp_input_data.read_data_sets(eeg_data_mat,
                                               FLAGS.fake_data,
                                               reshape_t=False,
                                               validation_size=896)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder, keep_prob = placeholder_inputs(
            FLAGS.batch_size, data_sets.feature_shape[3])
        # Build a Graph that computes predictions from the inference model.
        logits = autorun_infer.select_running_cnn(images_placeholder,
                                                  keep_prob,
                                                  layer=hyper_param['layer'],
                                                  feat=hyper_param['feat'],
                                                  cnn_id=model)
        # Add to the Graph the Ops for loss calculation.
        loss = rsvp_quick_cnn_model.loss(logits, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = rsvp_quick_cnn_model.training(loss, FLAGS.learning_rate)
        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = rsvp_quick_cnn_model.evaluation(logits, labels_placeholder)
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
                                                graph=sess.graph)
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
            if step % check_step == 0:
                # Print status to stdout.
                print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % check_step == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        logits,
                        images_placeholder,
                        labels_placeholder,
                        keep_prob,
                        data_sets.train,
                        csv_writer_acc,
                        csv_writer_auc)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        logits,
                        images_placeholder,
                        labels_placeholder,
                        keep_prob,
                        data_sets.validation,
                        csv_writer_acc,
                        csv_writer_auc)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        logits,
                        images_placeholder,
                        labels_placeholder,
                        keep_prob,
                        data_sets.test,
                        csv_writer_acc,
                        csv_writer_auc)

    # turn off writer after finish
    if csv_writer_acc is not None:
        csv_writer_acc.close()
    if csv_writer_auc is not None:
        csv_writer_auc.close()


def check_same_dict(x, y):

    if x['layer'] == y['layer']:
        if x['feat'] == y['feat']:
            return True
    return False


def def_hyper_param():

    hyper_param_list = []
    while len(hyper_param_list) < max_rand_search:
        replicated = False
        rnd_layer = random.choice(layer_list)
        rnd_feat = []
        # add the random features for each layer
        for layer_idx in range(0, rnd_layer):
            rnd_feat.append(random.choice(feat_list))
        # put them into dictionary
        hyper_param = {
            'layer':    rnd_layer,
            'feat':     rnd_feat
        }
        for hyper_param_element in hyper_param_list:
            # check if the element is replicated, if not, add to list
            if check_same_dict(hyper_param_element, hyper_param):
                replicated = True

        if not replicated:
            hyper_param_list.append(hyper_param)

    return hyper_param_list


def main(_):
    hyper_param_list = def_hyper_param()
    # hyper_param_list = [{'layer': 3, 'feat': [8, 16, 64]}]
# {'layer': 1, 'feat': [128]},
#                         {'layer': 2, 'feat': [128, 8]},
#                         {'layer': 2, 'feat': [128, 16]},
#                         {'layer': 2, 'feat': [128, 32]},
#

    # hyper_param_list = [{'layer': 3, 'feat': [4, 4, 512]},
    #                     {'layer': 3, 'feat': [8, 8, 512]},
    #                     {'layer': 3, 'feat': [4, 4, 256]},
    #                     {'layer': 4, 'feat': [4, 4, 4, 512]},
    #                     {'layer': 4, 'feat': [8, 8, 8, 512]},
    #                     {'layer': 4, 'feat': [4, 4, 4, 256]},
    #                     {'layer': 5, 'feat': [4, 4, 4, 4, 512]},
    #                     {'layer': 5, 'feat': [8, 8, 8, 8, 512]},
    #                     {'layer': 5, 'feat': [4, 4, 4, 4, 128]}]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

    models = [8]

    for model in models:
        for hyper_param in hyper_param_list:
            print("Currently running model: "+str(model))
            print("FeatMap: ")
            print(hyper_param['feat'])
            for idx in range(0, len(roi_property.DAT_TYPE_STR)):
            # for idx in range(3, 4):
                print("Data: " + roi_property.DAT_TYPE_STR[idx])
                for subIdx in range(0, 10):
                    print("Subject: " + str(subIdx))
                    orig_stdout, f = autorun_util.open_save_file(model, hyper_param['feat'], name_idx=idx, sub_idx=subIdx)
                    run_training(hyper_param, model, name_idx=idx, sub_idx=subIdx)
                    autorun_util.close_save_file(orig_stdout, f)

if __name__ == '__main__':
    tf.app.run()