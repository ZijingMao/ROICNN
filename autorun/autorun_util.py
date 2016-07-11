# -*- coding: utf-8 -*-
"""
Created on 3/8/16 6:24 PM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import roi_property

file_path = roi_property.SAVE_DIR

def str_name(name_idx):
    exp_type_str = roi_property.EXP_TYPE_STR[0]
    exp_name_str = roi_property.EXP_NAME_STR[0]
    dat_type_str = roi_property.DAT_TYPE_STR[name_idx]
    sub_str = roi_property.SUB_STR[0]
    chan_str = roi_property.CHAN_STR
    eeg_data = exp_type_str + '_' + \
               exp_name_str + '_' + \
               sub_str + '_' + \
               dat_type_str + '_' + \
               chan_str
    return eeg_data

acc_str = '.acc'
auc_str = '.auc'
log_str = '.log'
csv_str = '.csv'


def open_save_file(model, feat_list, log_flag=True, name_idx=0):
    if log_flag is False:
        return None, None

    name_str = str_name(name_idx)
    file_path_stub = file_path+name_str+'/'+str(model)+'/'
    if not os.path.exists(file_path_stub):
        os.makedirs(file_path_stub)

    file_name_stub = file_path_stub + 'feat_'+'_'.join(str(p) for p in feat_list)
    f_name = file_name_stub + log_str
    f = open(f_name, 'w')
    orig_stdout = sys.stdout
    sys.stdout = f

    return orig_stdout, f


def close_save_file(orig_stdout, f):
    if orig_stdout is not None:
        sys.stdout = orig_stdout
    if f is not None:
        f.close()


def csv_writer(model, feat_list, acc_flag=True, auc_flag=True, name_idx=0):

    file_path_stub = file_path+str_name(name_idx)+'/'+str(model)+'/'
    if not os.path.exists(file_path_stub):
        os.makedirs(file_path_stub)

    file_name_stub = file_path_stub + 'feat_'+'_'.join(str(p) for p in feat_list)
    csv_writer_acc_name = file_name_stub + acc_str + csv_str
    csv_writer_auc_name = file_name_stub + auc_str + csv_str

    csv_writer_acc = None
    csv_writer_auc = None
    # be careful no mistake here
    if acc_flag is True:
        csv_writer_acc = open(csv_writer_acc_name, 'w')
    if auc_flag is True:
        csv_writer_auc = open(csv_writer_auc_name, 'w')

    return csv_writer_acc, csv_writer_auc
