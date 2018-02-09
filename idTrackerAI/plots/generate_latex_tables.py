from __future__ import absolute_import, division, print_function
import os
import sys

import numpy as np
from pprint import pprint
import pandas as pd

def f_string(x):
    return x

def f_percentage(x):
    return '%.2f' %(x*100) if x != 1 else '%i' %(x*100)

def f_accuracy(x):
    return '%.3f' %(x*100) if x != 1 else '%i' %(x*100)

def f_time(x):
    return '%.2f' %x

def f_area(x):
    return '%i' %x

def f_float(x):
    return '%.2f' %x

def f_integer(x):
    return '%i' %x

def f_boolean(x):
    return 'yes' if x else 'no'

if __name__ == '__main__':

    path_to_tracked_videos_data_frame = '/media/chronos/ground_truth_results_backup/tracked_videos_data_frame.pkl'

    tracked_videos_data_frame = pd.read_pickle(path_to_tracked_videos_data_frame)

    columns_to_add = ['video_title',
                        'video_length_min',
                        'frame_rate',
                        'percentage_of_video_validated',
                        'mean_area_in_pixels',
                        'accuracy_identification_and_interpolation',
                        'accuracy_in_accumulation',
                        'accuracy_in_residual_identification_identification_and_interpolation',
                        'protocol_used',
                        'percentage_of_unoccluded_images',
                        'false_positive_rate_in_crossing_detector']

    new_columns_names = ['Video',
                'Length (min)',
                'Frame rate (fps)',
                'perc. of video reviewed',
                'Pixels per animal',
                'Accuracy',
                'Accuracy in accumulation and identification',
                'Accuracy in residual identification',
                'Protocol',
                'perc. of individual blobs',
                'False positive rate for DCD']

    formatters = [f_string,
                    f_time,
                    f_integer,
                    f_float,
                    f_area,
                    f_accuracy,
                    f_accuracy,
                    f_accuracy,
                    f_integer,
                    f_percentage,
                    f_percentage]

    ### short videos
    smaller_groups_videos_data_frame = tracked_videos_data_frame[tracked_videos_data_frame.number_of_animals <= 35]
    new_smaller = smaller_groups_videos_data_frame[columns_to_add].copy()
    new_smaller.columns = new_columns_names
    ### larger videos
    larger_groups_videos_data_frame = tracked_videos_data_frame[tracked_videos_data_frame.number_of_animals > 35]
    new_larger = larger_groups_videos_data_frame[columns_to_add].copy()
    new_larger.columns = new_columns_names

    with open('./plots/smaller_group_size.tex','w') as tf:
        tf.write(new_smaller.to_latex(index = False, formatters = formatters))
    with open('./plots/larger_group_size.tex','w') as tf:
        tf.write(new_larger.to_latex(index = False, formatters = formatters))
