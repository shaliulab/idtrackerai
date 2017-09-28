from __future__ import absolute_import, division, print_function
import os
import sys
from glob import glob
sys.path.append('./')
sys.path.append('./utils')
sys.path.append('./groundtruth_utils')

import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, is_color_like
import matplotlib
MARKERS = matplotlib.markers.MarkerStyle.markers.keys()[5:]
import seaborn as sns
import pandas as pd
from pprint import pprint
import time

from list_of_blobs import ListOfBlobs
from blob import Blob
from generate_light_groundtruth_blob_list import GroundTruth, GroundTruthBlob
from compute_statistics_against_groundtruth import get_statistics_against_groundtruth, \
                                                    compute_and_save_gt_accuracy

if __name__ == '__main__':

    path_to_ground_truth_hard_drive = '/media/rhea/ground_truth_results'
    folders_prefix = 'LargeGroups'
    experimental_group_folders = glob(path_to_ground_truth_hard_drive + '/*/')

    if os.path.isfile('../../Figures-paper/results_gt_accuracy.pkl'):
        print("results_gt_accuracy.pkl already exists \n")
        results_data_frame = pd.read_pickle('../../Figures-paper/results_gt_accuracy.pkl')
    else:
        print("results_gt_accuracy.pkl does not exist \n")
        results_data_frame = pd.DataFrame()

    for experimental_group_folder in experimental_group_folders:
        print("\n")
        print("experimental_group_folder ", experimental_group_folder)

        group_size_folders = glob(experimental_group_folder + '/*/')

        for group_size_folder in group_size_folders:
            print("group_size_folder ", group_size_folder)

            condition_folders = glob(group_size_folder + '/*/')

            for condition_folder in condition_folders:
                print("condition_folder ", condition_folder)

                condition = condition_folder[0].split('/')[-2]

                if os.path.isfile(os.path.join(condition_folder,'_groundtruth.npy')) \
                    and os.path.isdir(os.path.join(condition_folder,'session_gt')):

                    session_path = os.path.join(condition_folder,'session_gt')
                    video_object_path = os.path.join(session_path,'video_object.npy')
                    print("loading video object...")
                    video = np.load(video_object_path).item(0)

                    # check that gt_accuracy exists
                    if hasattr(video,'gt_accuracy'):
                        print('gt_accuracy ', video.gt_accuracy)
                    else:
                        compute_and_save_gt_accuracy(video_object_path, video)

                results_data_frame = results_data_frame.append({'date': time.strftime("%c"),
                                                                'video_name': video.video_path,
                                                                'height': video._height,
                                                                'width': video._width,
                                                                'num_frames': video.number_of_frames,
                                                                'frames_per_second': video._frames_per_second,
                                                                'number_of_animals': video._number_of_animals,
                                                                'preprocessing_type': video._preprocessing_type,
                                                                'subtract_bkg': video.subtract_bkg,
                                                                'apply_ROI': video.apply_ROI,
                                                                'min_threshold': video._min_threshold,
                                                                'max_threshold': video._max_threshold,
                                                                'min_area': video._min_area,
                                                                'max_area': video._max_area,
                                                                'reduce_resolution': None,
                                                                'resolution_reduction': None,
                                                                'crossing_detector_used': None,
                                                                'maximum_number_of_blobs': video._maximum_number_of_blobs,
                                                                'median_body_length': None,
                                                                'portrait_size': None,
                                                                'individual_fragments_lenghts': None,
                                                                'individual_fragments_distance_travelled': None,
                                                                'crossing_fragments_lengths': None,
                                                                'number_of_crossings': None,
                                                                'number_of_global_fragments': None,
                                                                'number_of_accumulated_global_fragments': None,
                                                                'number_of_non_certain_global_fragments': None,
                                                                'number_of_randomly_assigned_global_fragments': None,
                                                                'number_of_nonconsistent_global_fragments': None,
                                                                'number_of_nonunique_global_fragments': None,
                                                                'validation_accuracy': None,
                                                                'validation_individual_accuracies': None,
                                                                'overall_P2': None,
                                                                'ratio_of_accumulated_images': None,
                                                                'gt_start_end':video.gt_start_end,
                                                                'gt_accuracy':video.gt_accuracy,
                                                                'gt_individual_accuracy':video.gt_individual_accuracy,
                                                                'gt_accuracy_assigned':video.gt_accuracy_assigned,
                                                                'gt_individual_accuracy_assigned':video.gt_individual_accuracy_assigned,
                                                                'crossing_detector_accuracy': None,
                                                                'time_preprocessing': None,
                                                                'number_of_accumulation_trials_before_pretraining': None,
                                                                'time_accumulation_before_pretraining': None,
                                                                'pretraining_used': None,
                                                                'time_pretraining': None,
                                                                'number_of_accumulation_trials_after_pretraining': None,
                                                                'time_accumulation_after_pretraining': None,
                                                                'time_assignment': None,
                                                                'time_postprocessing': None,
                                                                'total_time': None,
                                                                 }, ignore_index=True)

                results_data_frame.to_pickle('../Figures-paper/results_data_frame.pkl')




    #
    # # plot
    # plt.ion()
    # sns.set_style("ticks")
    # fig, ax_arr = plt.subplots(2,2, sharex = True)
    # window = plt.get_current_fig_manager().window
    # screen_y = window.winfo_screenheight()
    # screen_x = window.winfo_screenwidth()
    # fig.set_size_inches((screen_x*2/3/100,screen_y/100))
    # fig.suptitle('Single image identification accuracy - library %s - %i repetitions' %('G',
    #                                                 len(results_data_frame['repetition'].unique())), fontsize = 25)
    #
    #
    #
    #
    # plt.minorticks_off()
    #
    # # plt.show()
    # fig.savefig('single_image_identification_accuracy.pdf', transparent=True)
