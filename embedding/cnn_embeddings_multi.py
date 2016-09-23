# -*- coding: utf-8 -*-
"""
Created on 3/26/16 10:16 AM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys

import autorun_infer
import autorun_util
import rsvp_input_data
import rsvp_quick_cnn_model
from workproperty import roi_property
import sklearn.metrics as metrics
import numpy as np
import scipy.io

import tensorflow as tf
from rsvp_quick_inference import set_batch_size as set_batch_infer_size
import h5py

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
max_step = 5000    # to guarantee 64 epochs # should be training sample_size
check_step = max_step/50
layer_list = roi_property.LAYER_LIST
feat_list = roi_property.FEAT_LIST
max_rand_search = roi_property.MAX_RAND_SEARCH

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', max_step, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', roi_property.BATCH_SIZE, 'Batch size. Must divide evenly into the dataset sizes.')
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
    images_feed, labels_feed = data_set.next_batch_no_shuffle(FLAGS.batch_size)
    feed_dict = {
        images_pl: images_feed,
        keep_prob: drop_rate
    }
    return feed_dict, labels_feed


def do_eval(sess,
            keep_prob,
            data_set,
            name):
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
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    true_label = np.array([]).reshape(0,)   # the label information is only 1 dimension
    true_feat = np.array([]).reshape((0, 64, 64, 8))   # the feature information is 4 dimensions
    # true_feat = np.array([]).reshape((0, 128))

    for step in xrange(steps_per_epoch):
        images_feed, labels_feed = data_set.next_batch_no_shuffle(FLAGS.batch_size)
        cnn_tensor = sess.graph.get_tensor_by_name('conv1/conv1:0')
        # cnn_tensor = sess.graph.get_tensor_by_name('local2/local2:0')
        forward_feats = sess.run(cnn_tensor, {'Placeholder:0': images_feed, keep_prob: 1})
        forward_labels = labels_feed          # define the labels output

        true_label = np.concatenate((true_label, forward_labels), axis=0)
        true_feat = np.concatenate((true_feat, forward_feats), axis=0)
        string_ = str(step) + ' / ' + str(num_examples)
        sys.stdout.write("\r%s" % string_)
        sys.stdout.flush()

    # now you can save the feature, matlab save in mat file
    scipy.io.savemat(roi_property.SAVE_DIR+'feature_output/'+name+'.mat',
                     mdict={name+'_x': true_feat, name+'_y': true_label})

    print('Feature saved')


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
    # Get the sets of images and labels for training, validation, and
    # test on RSVP.
    eeg_data = autorun_util.str_name(name_idx, sub_idx)
    eeg_data_dir = roi_property.FILE_DIR + \
                   'rsvp_data/mat_sub/' + eeg_data
    eeg_data_mat = eeg_data_dir + '.mat'
    data_sets = rsvp_input_data.read_all_data(eeg_data_mat)
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

        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        do_eval(sess,
                keep_prob,
                data_sets.train,
                'train')
        do_eval(sess,
                keep_prob,
                data_sets.test,
                'test')

        return


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
    # hyper_param_list = def_hyper_param()
    hyper_param_list = [{'layer': 2, 'feat': [32, 8]}]
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
    #                     {'layer': 5, 'feat': [4, 4, 4, 4, 128]}]

    models = [1]

    for model in models:
        for hyper_param in hyper_param_list:
            print("Currently running model: "+str(model))
            print("FeatMap: ")
            print(hyper_param['feat'])
            # for idx in range(3, len(roi_property.DAT_TYPE_STR)):
            for idx in range(5, 6):
                print("Data: " + roi_property.DAT_TYPE_STR[idx])
                for subIdx in range(10, 11):
                    print("Subject: " + str(subIdx))
                    # orig_stdout, f = autorun_util.open_save_file(model, hyper_param['feat'], name_idx=idx, sub_idx=subIdx)
                    run_training(hyper_param, model, name_idx=idx, sub_idx=subIdx)
                    # autorun_util.close_save_file(orig_stdout, f)

if __name__ == '__main__':
    tf.app.run()