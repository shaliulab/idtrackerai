from __future__ import absolute_import, division, print_function
import os
import sys

import numpy as np
from pprint import pprint
import pandas as pd
import time

def f_string(x):
    return x

def f_accuracy(x):
    return '%.3f' %(x*100) if x != 1 else '(%i)' %(x*100)

def f_individual_accuracy(x):
    return '/%.3f' %(x*100) if x != 1 else '/%i' %(x*100)

def f_float_parentesis(x):
    return '(%.2f)' %x if x != 1 else '(%i)' %x

def f_crossings(x):
    return '(%i)' %x

def f_percentage(x):
    return '%.2f' %(x*100) if x != 1 else '%i' %(x*100)

def f_time(x):
    return time.strftime("%H:%M:%S", time.gmtime(x))

def f_area(x):
    return '%i' %x

def f_float(x):
    return '%.2f' %x if x != 100 else '%i' %x

def f_integer(x):
    return '%i' %x

def f_boolean(x):
    return 'yes' if x else 'no'

def write_latex_table_for_subset_dataframe(tracked_videos_folder, data_frame, columns_to_include, subset_condition, subtable_name):
    new_columns_names = list(zip(*columns_to_include)[1])
    formatters = list(zip(*columns_to_include)[2])
    columns_to_include = list(zip(*columns_to_include)[0])

    subset_data_frame = data_frame[subset_condition]
    print(subset_data_frame[["animal_type", "number_of_animals", "video_name", "percentage_of_video_validated"]])
    subset_data_frame = subset_data_frame[columns_to_include].copy()
    subset_data_frame.columns = new_columns_names
    latex_table_name = subtable_name + '.tex'
    latex_table_path = os.path.join(tracked_videos_folder, latex_table_name)
    with open(latex_table_path,'w') as file:
        file.write(subset_data_frame.to_latex(index = False, formatters = formatters))

if __name__ == '__main__':
    path_to_results_hard_drive = '/media/chronos/ground_truth_results_backup'
    tracked_videos_folder = os.path.join(path_to_results_hard_drive, 'tracked_videos')
    path_to_tracked_videos_data_frame = os.path.join(tracked_videos_folder, 'tracked_videos_data_frame.pkl')

    tracked_videos_data_frame = pd.read_pickle(path_to_tracked_videos_data_frame)

    columns_to_include = [('video_title','Video', f_string),
                        ('video_length_sec', 'Duration', f_time),
                        ('frame_rate', 'Frame rate (fps)', f_integer),
                        ('mean_area_in_pixels', 'Pixels per animal', f_area),
                        ('protocol_used', 'Protocol', f_integer),
                        ('number_of_crossing_fragments_in_validated_part', 'Number of crossings reviewed', f_integer),
                        ('accuracy_in_accumulation', 'Accuracy accum. identif.', f_accuracy),
                        ('accuracy_identification_and_interpolation', 'Accuracy indiv. images', f_accuracy),
                        ('individual_accuracy_interpolated', 'Individual accuracy', f_individual_accuracy),
                        ('rate_nonidentified_animals_indentification_and_interpolation','perc. not identified', f_accuracy),
                        ('rate_misidentified_animals_identification_and_interpolation', 'perc. misidentified', f_accuracy)]

    ### smaller groups videos
    print("generating smaller groups table")
    condition = [x and not y for (x,y) in zip(list(tracked_videos_data_frame.number_of_animals < 35), list(tracked_videos_data_frame.bad_example))]
    # condition = tracked_videos_data_frame.number_of_animals <= 35 and not tracked_videos_data_frame.idTracker_video
    write_latex_table_for_subset_dataframe(tracked_videos_folder, tracked_videos_data_frame, columns_to_include, condition, 'smaller_group_sizes_table')
    ### larger groups videos
    print("generating larger groups videos table")
    condition = [x and not y for (x,y) in zip(list(tracked_videos_data_frame.number_of_animals >= 35), list(tracked_videos_data_frame.bad_example))]
    write_latex_table_for_subset_dataframe(tracked_videos_folder, tracked_videos_data_frame, columns_to_include, condition, 'larger_group_sizes_table')
    ### idTracker videos
    print("generating bad videos table")
    condition = [bool(x) for x in tracked_videos_data_frame.bad_example]
    write_latex_table_for_subset_dataframe(tracked_videos_folder, tracked_videos_data_frame, columns_to_include, condition, 'bad_videos_table')
