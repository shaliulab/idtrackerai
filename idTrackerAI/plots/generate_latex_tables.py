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

def f_individual_accuracy(x):
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
    path_to_results_hard_drive = '/media/atlas/ground_truth_results_backup'
    tracked_videos_folder = os.path.join(path_to_results_hard_drive, 'tracked_videos')
    path_to_tracked_videos_data_frame = os.path.join(tracked_videos_folder, 'tracked_videos_data_frame.pkl')

    tracked_videos_data_frame = pd.read_pickle(path_to_tracked_videos_data_frame)

    columns_to_include = ['video_title',
                        'video_length_min',
                        'frame_rate',
                        'mean_area_in_pixels',
                        'protocol_used',
                        'percentage_of_video_validated',
                        'number_of_crossing_fragments_in_validated_part',
                        'accuracy_identification_and_interpolation',
                        'individual_accuracy_interpolated',
                        'accuracy_in_accumulation',
                        'accuracy_in_residual_identification_identification_and_interpolation',
                        'percentage_of_unoccluded_images',
                        'false_positive_rate_in_crossing_detector']

    new_columns_names = ['Video',
                'Length (min)',
                'Frame rate (fps)',
                'Pixels per animal',
                'Protocol',
                'perc. of video reviewed',
                'Number of crossings',
                'Accuracy',
                'Individual accuracy',
                'Accuracy in accumulation and identification',
                'Accuracy in residual identification',
                'perc. of individual blobs',
                'False positive rate for DCD']

    formatters = [f_string,
                    f_time,
                    f_integer,
                    f_integer,
                    f_integer,
                    f_float,
                    f_area,
                    f_accuracy,
                    f_individual_accuracy,
                    f_accuracy,
                    f_accuracy,
                    f_percentage,
                    f_percentage]

    def write_latex_table_for_subset_dataframe(tracked_videos_folder, data_frame, columns_to_include, new_columns_names, formatters, subset_condition, subtable_name):
        assert len(columns_to_include) == len(new_columns_names) == len(formatters)

        subset_data_frame = data_frame[subset_condition]
        subset_data_frame = subset_data_frame[columns_to_include].copy()
        subset_data_frame.columns = new_columns_names
        latex_table_name = subtable_name + '.tex'
        latex_table_path = os.path.join(tracked_videos_folder, latex_table_name)
        with open(latex_table_path,'w') as file:
            file.write(subset_data_frame.to_latex(index = False, formatters = formatters))

    ### smaller groups videos
    print("generating smaller groups table")
    condition = [x and not y for (x,y) in zip(list(tracked_videos_data_frame.number_of_animals <= 35), list(tracked_videos_data_frame.idTracker_video))]
    # condition = tracked_videos_data_frame.number_of_animals <= 35 and not tracked_videos_data_frame.idTracker_video
    write_latex_table_for_subset_dataframe(tracked_videos_folder, tracked_videos_data_frame, columns_to_include, new_columns_names, formatters, condition, 'smaller_group_sizes_table')
    ### larger groups videos
    print("generating larger groups videos table")
    condition = [x and not y for (x,y) in zip(list(tracked_videos_data_frame.number_of_animals > 35), list(tracked_videos_data_frame.idTracker_video))]
    write_latex_table_for_subset_dataframe(tracked_videos_folder, tracked_videos_data_frame, columns_to_include, new_columns_names, formatters, condition, 'larger_group_sizes_table')
    ### idTracker videos
    print("generating idTracker videos table")
    condition = [bool(x) for x in tracked_videos_data_frame.idTracker_video]
    write_latex_table_for_subset_dataframe(tracked_videos_folder, tracked_videos_data_frame, columns_to_include, new_columns_names, formatters, condition, 'idTracker_videos_table')
