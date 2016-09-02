# Created by Ehren Biglari at 5/25/2016
# base on code written by Zijing Mao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

import autorun_deconv
import autorun_util
import rsvp_input_data
import rsvp_quick_cnn_model
from workproperty import roi_property
import sklearn.metrics as metrics
import numpy as np
import tensorflow as tf
from rsvp_quick_inference import set_batch_size as set_batch_infer_size
import scipy.io as spio

EXP_TYPE_STR = roi_property.EXP_TYPE_STR[0]
EXP_NAME_STR = roi_property.EXP_NAME_STR[0]
DAT_TYPE_STR = roi_property.DAT_TYPE_STR[5]
SUB_STR = roi_property.SUB_STR[0]
CHAN_STR = roi_property.CHAN_STR

EEG_DATA = EXP_TYPE_STR + '_' + \
           EXP_NAME_STR + '_' + \
           SUB_STR + '_' + \
           DAT_TYPE_STR + '_' + \
           CHAN_STR
EEG_DATA_DIR = roi_property.FILE_DIR + \
               'rsvp_data/mat_sub/' + EEG_DATA
EEG_TF_DIR = roi_property.FILE_DIR + \
               'rsvp_data/' + EEG_DATA
EEG_DATA_MAT = EEG_DATA_DIR + '.mat'

# Basic model parameters as external flags.
# TODO try to change learning rate in the rsvp folder

learning_rate = 0.03
learn_rate_decay_factor = 0.99997
decay_steps = 100
choose_cnn_type = 1
#deconv_batch_size = 1
batch_size = 125
max_step = 200   #roi_property.MEDIUM_TRAIN_SIZE    # to guarantee 64 epochs # should be training sample_size
check_step = 100

mode = autorun_deconv.TEST

layer_list = roi_property.LAYER_LIST
feat_list = roi_property.FEAT_LIST
max_rand_search = roi_property.MAX_RAND_SEARCH

flags = tf.app.flags
FLAGS = flags.FLAGS
lr = 0
flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', max_step, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', batch_size, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', roi_property.WORK_DIR + 'data/rsvp_train/', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                                         'for unit testing.')



def set_batch_size(_batch_size):
    global batch_size
    set_batch_infer_size(_batch_size)
    batch_size = _batch_size


def define_learning_rate():
    global global_step, lr, flags
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    lr = tf.train.exponential_decay(learning_rate,
                                    global_step,
                                    decay_steps,
                                    learn_rate_decay_factor,
                                    staircase=True)

    #flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
    return


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
                                                           rsvp_quick_cnn_model.IMAGE_SIZE,
                                                           rsvp_quick_cnn_model.IMAGE_SIZE,
                                                           1))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    keep_prob = tf.placeholder(tf.float32)

    return images_placeholder, labels_placeholder, keep_prob


def placeholder_inputs2(batch_size):
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
                                                           1))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    keep_prob = tf.placeholder(tf.float32)

    filter_num = tf.placeholder(tf.int64)

    image_num = tf.placeholder(tf.int64)

    max_act_pl = tf.placeholder(tf.float32)

    max_ind_pl = tf.placeholder(tf.int64)

    return images_placeholder, labels_placeholder, keep_prob, filter_num, image_num, max_act_pl, max_ind_pl


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
    global batch_size

    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.

    images_feed, labels_feed = data_set.next_batch(batch_size, FLAGS.fake_data)

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        keep_prob: drop_rate,
    }
    return feed_dict

def fill_feed_dict2(data_set, drop_rate, images_pl, labels_pl, keep_prob,
                       filter_num, image_num, max_act_pl, max_ind_pl,
                       filter_num_val=0, image_num_val=0, max_act_val=0.0, max_ind_val=0,
                       batch_offset=0):

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
    global batch_size

    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.

    images_feed, labels_feed = data_set.next_batch_at_offset(batch_size, batch_offset)

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        keep_prob: drop_rate,
        filter_num: int(filter_num_val),
        image_num: int(image_num_val),
        max_act_pl: max_act_val,
        max_ind_pl: int(max_ind_val)
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
    global batch_size
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size

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


def run_training(hyper_param, model):
    global batch_size
    '''
    Train RSVP for a number of steps.
    Args:
        hyper_param: three elements, layer & feat & model
        model:

    Returns:

    '''
    # initialize the summary to write
    csv_writer_acc, csv_writer_auc = autorun_util.csv_writer(model, hyper_param['feat'])
    # Get the sets of images and labels for training, validation, and
    # test on RSVP.

    set_batch_size(125)

    data_sets = rsvp_input_data.read_data_sets(EEG_DATA_MAT,
                                               FLAGS.fake_data,
                                               reshape_t=False)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder, keep_prob = \
                                                placeholder_inputs(batch_size)
        # define_learning_rate()
        # Build a Graph that computes predictions from the inference model.
        results = autorun_deconv.select_running_cnn(images_placeholder,
                                                keep_prob, None,
                                                None, None, None, None,
                                                mode=0,
                                                layer=hyper_param['layer'],
                                                feat=hyper_param['feat'],
                                                cnn_id=model
                                                )

        logits = results

        # Add to the Graph the Ops for loss calculation.
        loss = rsvp_quick_cnn_model.loss(logits, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = rsvp_quick_cnn_model.training(loss, lr)
        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = rsvp_quick_cnn_model.evaluation(logits, labels_placeholder)
        # Build the summary operation based on the TF collection of Summaries.
        #summary_op = tf.merge_all_summaries()
        # Create a saver for writing training checkpoints.

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.


        # Instantiate a SummaryWriter to output summaries and the Graph.
        #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
        #                                        graph_def=sess.graph_def)

        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver()

        saver.restore(sess, "/home/caffe1/PycharmProjects/ROICNN/data/rsvp_train/-4999")
        print("Model restored.")

        # And then after everything is built, start the training loop.
        for step in xrange(FLAGS.max_steps):

            start_time = time.time()
            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            if mode == autorun_deconv.TEST:
                feed_dict = fill_feed_dict(data_sets.train,
                                           0.5, #0.0625,  # was .5 works good
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
                #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                #summary_writer.add_summary(summary_str, step)
            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % check_step == 0 or (step + 1) == FLAGS.max_steps:
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
                # print('Validation Data Eval:')
                #do_eval(sess,
                #        eval_correct,
                #        logits,
                #        images_placeholder,
                #        labels_placeholder,
                #        keep_prob,
                #        filter_num, image_num,
                #        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        logits,
                        images_placeholder,
                        labels_placeholder,
                        keep_prob,
                        data_sets.test)


	    # Change batchsize and model to fit new batchsize
        temp = set(tf.all_variables())

        set_batch_size(1)

        images_placeholder2, labels_placeholder2, keep_prob2, filter_num2, image_num2, max_act_pl2, max_ind_pl2 = \
            placeholder_inputs2(batch_size)

        # Build a Graph that computes predictions from the inference model.
        returnTensors = autorun_deconv.select_running_cnn(images_placeholder2,
                                                          keep_prob2, None,
                                                          filter_num2, image_num2, max_act_pl2, max_ind_pl2,
                                                          mode=autorun_deconv.FIND_MAX_ACTIVATION_GET_SWITCHES,
                                                          layer=hyper_param['layer'],
                                                          feat=hyper_param['feat'],
                                                          cnn_id=model)

        feed_dict2 = fill_feed_dict2(data_sets.train,
                                    1.0,
                                    images_placeholder2,
                                    labels_placeholder2,
                                    keep_prob2,
                                    filter_num2, image_num2, max_act_pl2, max_ind_pl2)

        # writer = tf.train.SummaryWriter("/home/e/deconvgraph/deconv_logs", sess.graph)

        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        updates = autorun_deconv.get_update_threshold()
        clear_vars = autorun_deconv.get_clear_variables()

        # Find maximum activation
        num_layers = hyper_param['layer']

        top_nth = []
        feat_num = []
        max_ind_list = []
        max_image_ind_list = []
        max_activation_list = []
        max_labels_list = []

        #select the nth highest feature
        for n in range(30):
            for step in xrange(10000):
                print('test pass' + str(step))
                feed_dict2 = fill_feed_dict2(data_sets.train,
                                            1.0,
                                            images_placeholder2,
                                            labels_placeholder2,
                                            keep_prob2,
                                            filter_num2, image_num2, max_act_pl2, max_ind_pl2,
                                            0, step, 0.0, 0, # placeholders
                                            batch_offset=step)  # batch offset

                returnTensorVals = sess.run(returnTensors, feed_dict=feed_dict2)
                _ = returnTensorVals
                _ = returnTensorVals

            # unpack results
            max_ind = returnTensorVals[-3 * num_layers:-2*num_layers]
            max_image_ind = returnTensorVals[-2*num_layers:-num_layers]
            max_activation = returnTensorVals[-num_layers:]

            # Convert to python lists
            if n==0:
                for l in range(0, num_layers):
                    list_len = len(max_ind[l].tolist())
                    top_nth.append([n]*list_len)
                    feat_num.append(range(list_len))
                    max_ind_list.append(max_ind[l].tolist())
                    max_image_ind_list.append(max_image_ind[l].tolist())
                    max_activation_list.append(max_activation[l].tolist())
            else:
                for l in range(0, num_layers):
                    list_len = len(max_ind[l].tolist())
                    top_nth[l].extend([n]*list_len)
                    feat_num[l].extend(range(list_len))
                    max_ind_list[l].extend(max_ind[l].tolist())
                    max_image_ind_list[l].extend(max_image_ind[l].tolist())
                    max_activation_list[l].extend(max_activation[l].tolist())

            # update threshold
            returnTensorsTmp = list(returnTensors)
            returnTensorsTmp.extend(updates)
            # update max threshold
            sess.run(returnTensorsTmp, feed_dict=feed_dict2)

            # restart at batch zero
            feed_dict2 = fill_feed_dict2(data_sets.train,
                                         1.0,
                                         images_placeholder2,
                                         labels_placeholder2,
                                         keep_prob2,
                                         filter_num2, image_num2, max_act_pl2, max_ind_pl2,
                                         0, 0, 0.0, 0,  # placeholders
                                         batch_offset=0)  # batch offset

            # clear data
            returnTensorsTmp = list(returnTensors)
            returnTensorsTmp.extend(clear_vars)
            # update max threshold
            sess.run(returnTensorsTmp, feed_dict=feed_dict2)


        for l in range(0, num_layers):
            max_labels_np_tmp = data_sets.train.get_labels(max_image_ind_list[l])
            max_labels_list.append(max_labels_np_tmp.tolist())



        # find the top 9 activations from all features in top layer
        cur_layer = 1
        max_activation_info = zip(max_labels_list[cur_layer], max_activation_list[cur_layer], max_image_ind_list[cur_layer], max_ind_list[cur_layer],
                                  feat_num[cur_layer], top_nth[cur_layer])

        sorted_activations_neg = sorted(max_activation_info, key=lambda x: (x[0], -x[1]))

        sorted_activations_pos = sorted(max_activation_info, key=lambda x: (-x[0], -x[1]))


        # Reconstruct

        temp = set(tf.all_variables())

        top_nth_reconstructions = []
        layer_switches = []

        temp = set(tf.all_variables())

        returnTensors = autorun_deconv.select_running_cnn(images_placeholder2,
                                                          keep_prob2, cur_layer,
                                                          filter_num2, image_num2,
                                                          max_act_pl2, max_ind_pl2,
                                                          mode=autorun_deconv.RECONSTRUCT_INPUT,
                                                          layer=hyper_param['layer'],
                                                          feat=hyper_param['feat'],
                                                          cnn_id=model
                                                          )
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        # top 9 negative label reconstructions
        input_reconstructions = []
        input_images = []
        batch_nums = []
        max_acts = []
        max_indicies = []
        max_filters = []
        for top_nth in range(0, 30):
            _, max_act_val, batch_num, max_ind_val, filter_num_val, _ = sorted_activations_neg[top_nth]
            feed_dict2 = fill_feed_dict2(data_sets.train,
                                        1.0,
                                        images_placeholder2,
                                        labels_placeholder2,
                                        keep_prob2,
                                        filter_num2, image_num2, max_act_pl2, max_ind_pl2,  # placeholders
                                        filter_num_val, 0,  max_act_val, max_ind_val,  # values
                                        batch_num)  # batch offset

            image = feed_dict2[images_placeholder2].copy()
            max_acts.append(max_act_val)
            max_indicies.append(max_ind_val)
            max_filters.append(filter_num_val)
            batch_nums.append(batch_num)
            input_images.append(image)  # unpack
            returnTensorVals = sess.run(returnTensors, feed_dict=feed_dict2)
            input_reconstructions.append(returnTensorVals[2])   # unpack


        top_nth_reconstructions_np = np.array([input_reconstructions])
        top_nth_images_np = np.array([input_images])

        spio.savemat('/home/e/deconvresults/neg.mat',
                     dict(recon = top_nth_reconstructions_np, images = top_nth_images_np,
                          batch_nums=batch_nums, max_acts=max_acts,max_indicies=max_indicies,
                          max_filters=max_filters))

        # top 9 positive label reconstructions
        input_reconstructions = []
        input_images = []
        batch_nums = []
        max_acts = []
        max_indicies = []
        max_filters = []
        for top_nth in range(0, 30):
            _, max_act_val, batch_num, max_ind_val, filter_num_val, _ = sorted_activations_pos[top_nth]
            feed_dict2 = fill_feed_dict2(data_sets.train,
                                         1.0,
                                         images_placeholder2,
                                         labels_placeholder2,
                                         keep_prob2,
                                         filter_num2, image_num2, max_act_pl2, max_ind_pl2,  # placeholders
                                         filter_num_val, 0, max_act_val, max_ind_val,  # values
                                         batch_num)  # batch offset

            image = feed_dict2[images_placeholder2].copy()
            max_acts.append(max_act_val)
            max_indicies.append(max_ind_val)
            max_filters.append(filter_num_val)
            batch_nums.append(batch_num)
            input_images.append(image)  # unpack
            returnTensorVals = sess.run(returnTensors, feed_dict=feed_dict2)
            input_reconstructions.append(returnTensorVals[2])  # unpack


        top_nth_reconstructions_np = np.array([input_reconstructions])
        top_nth_images_np = np.array([input_images])

        spio.savemat('/home/e/deconvresults/pos.mat',
                     dict(recon=top_nth_reconstructions_np, images=top_nth_images_np, batch_nums=batch_nums, max_acts=max_acts,max_indicies=max_indicies,max_filters=max_filters))




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
    #hyper_param_list = def_hyper_param()
    hyper_param_list = [{'layer': 2, 'feat': [32, 64]}]

    for model in range(0, 1):
        for hyper_param in hyper_param_list:
            print("Currently running: ")
            print("FeatMap: ")
            print(hyper_param['feat'])
            print("Model" + str(model))
            orig_stdout, f = autorun_util.open_save_file(model, hyper_param['feat'])
            run_training(hyper_param, model)
            autorun_util.close_save_file(orig_stdout, f)

if __name__ == '__main__':
    tf.app.run()
