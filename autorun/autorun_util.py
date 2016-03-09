# -*- coding: utf-8 -*-
"""
Created on 3/8/16 6:24 PM
@author: Zijing Mao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import roi_property
import autorun_rsvp_roi_cnn_64_chan

file_path = roi_property.SAVE_DIR
name_str = autorun_rsvp_roi_cnn_64_chan.EEG_DATA

acc_str = '.acc'
auc_str = '.auc'
log_str = '.log'
csv_str = '.csv'

orig_stdout = sys.stdout
f = file('out.txt', 'w')


def open_save_file():
    sys.stdout = f


def close_save_file():
    sys.stdout = orig_stdout
    f.close()