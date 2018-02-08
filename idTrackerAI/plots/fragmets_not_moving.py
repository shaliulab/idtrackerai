from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./')
sys.path.append('./utils')
sys.path.append('./network/identification_model')

import numpy as np
from matplotlib import pyplot as plt

from fragment import Fragment
from list_of_fragments import ListOfFragments
from list_of_global_fragments import ListOfGlobalFragments
from GUI_utils import selectDir

def get_frames_and_frames_moving_for_fragments(fragments):
    frames_moving = []
    frames = []
    for f in fragments:
        frames.append(f.number_of_images)
        velocities = f.frame_by_frame_velocity()
        frames_moving.append(sum(np.log10(velocities) > moving_threshold))
    frames_moving = [f for f in frames_moving if f > 0]
    return frames, frames_moving

if __name__ == '__main__':
    session_paths = ['/media/themis/ground_truth_results_backup/10_fish_group4/first/session_20180122',
             '/media/themis/ground_truth_results_backup/10_fish_group5/first/session_20180131',
             '/media/themis/ground_truth_results_backup/10_fish_group6/first/session_20180202',
             '/media/themis/ground_truth_results_backup/38 drosophila (females males)/Canton_N38_top_video_01-31-18_10-50-14/session_20180201',
             '/media/themis/ground_truth_results_backup/72 drosophila (females - males)/session_20180201',
             '/media/themis/ground_truth_results_backup/80 drosophila (females males)/Canton_N80_11-28-17_17-21-32/session_20180123',
             '/media/themis/ground_truth_results_backup/ants_andrew_1/session_20180206',
             '/media/themis/ground_truth_results_backup/idTrackerDeep_LargeGroups_1/100/First/session_20180102',
             '/media/themis/ground_truth_results_backup/idTrackerDeep_LargeGroups_1/60/First/session_20180108',
             '/media/themis/ground_truth_results_backup/idTrackerDeep_LargeGroups_2/TU20170307/numberIndivs_100/First/session_20180104',
             '/media/themis/ground_truth_results_backup/idTrackerDeep_LargeGroups_2/TU20170307/numberIndivs_60/First/session_20171221',
             '/media/themis/ground_truth_results_backup/idTrackerDeep_LargeGroups_3/100fish/First/session_02122017',
             '/media/themis/ground_truth_results_backup/idTrackerDeep_LargeGroups_3/60fish/First/session_20171225',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Hipertec_pesados/Medaka/2012may31/Grupo10/session_20180201',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Hipertec_pesados/Medaka/2012may31/Grupo5/session_20180131',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Hipertec_pesados/Medaka/20fish_contapa/session_20180201',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Moscas/2011dic12/Video_4fem_2mal_bottom/session_20180130',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Moscas/20121010/PlatoGrande_8females_2/session_20180131',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/NatureMethods/Isogenicos/Wik_8_grupo4/session_20180130',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/NatureMethods/Ratones4/session_20180205',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/NatureMethods/VideoRatonesDespeinaos3/session_20180206',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Ratones/20121203/2aguties/session_20180204',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Ratones/20121203/2negroscanosos/session_20180204',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Ratones/20121203/2negroslisocanoso/session_20180205',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Ratones/20121203/2negroslisos/session_20180205',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/ValidacionTracking/Moscas/Platogrande_8females/session_20180131',
             '/media/themis/ground_truth_results_backup/idTrackerVideos/Zebrafish_nacreLucie/pair3ht/session_20180207']

    for session_path in session_paths:
        print("\nsession_path: ", session_path)
        if 'drosophila' in session_path\
            or ('Moscas' in session_path and 'PlatoGrande' in session_path)\
            or ('Moscas' in session_path and 'Platogrande' in session_path):

            print("loading video")
            video = np.load(os.path.join(session_path, 'video_object.npy')).item()
            print("loading fragments")
            list_of_fragments = np.load(os.path.join(session_path, 'preprocessing', 'fragments.npy')).item()
            print("loading global_fragments")
            list_of_global_fragments = ListOfGlobalFragments.load(os.path.join(session_path, 'preprocessing', 'global_fragments.npy'),
                                            list_of_fragments.fragments)

            ### Overall distribution of velocities
            nbins = 100
            vels_good = np.hstack([f.frame_by_frame_velocity() for f in list_of_fragments.fragments if f.is_an_individual])
            vels_good = vels_good[vels_good != 0]
            min = np.min(vels_good)
            max = np.max(vels_good)
            logbins = np.linspace(np.log10(min), np.log10(max), nbins)

            plt.ion()
            fig, ax_arr = plt.subplots(1,1)
            fig.suptitle(session_path)
            ax_arr.hist(np.log10(vels_good), bins = logbins, normed = True)
            moving_threshold = 0

            ### Compute number of frames number_of_frames
            frames_good, frames_moving_good = get_frames_and_frames_moving_for_fragments(list_of_fragments.fragments)

            nbins = 25
            min = 1
            max = np.max(frames_good)
            logbins = np.linspace(np.log10(min), np.log10(max), nbins)

            fig, ax_arr = plt.subplots(1,2, sharex = True, sharey = True)
            ax_arr[00].set_title('good')
            ax_arr[0].hist(np.log10(frames_good), bins = logbins, normed = True)
            ax_arr[1].hist(np.log10(frames_moving_good), bins = logbins, normed = True)

            ### Distribution of frames in the fragments of the first global fragment
            list_of_global_fragments.order_by_distance_travelled()
            accumulation_trial = int(video.accumulation_folder[-1])
            global_fragment_index_used_for_accumulation_good = accumulation_trial if accumulation_trial == 0 else accumulation_trial - 1
            first_global_fragment_good = list_of_global_fragments.global_fragments[global_fragment_index_used_for_accumulation_good]

            frames_good, frames_moving_good = get_frames_and_frames_moving_for_fragments(first_global_fragment_good.individual_fragments)

            nbins = 25
            min = 1
            max = np.max(frames_good)
            logbins = np.linspace(np.log10(min), np.log10(max), nbins)

            fig, ax_arr = plt.subplots(1,2, sharex = True, sharey = True)
            ax_arr[0].set_title('good')
            ax_arr[0].hist(np.log10(frames_good), bins = logbins, normed = True)
            ax_arr[1].hist(np.log10(frames_moving_good), bins = logbins, normed = True)

            video.minimum_number_of_frames_moving_in_first_global_fragment = np.min(frames_moving_good)

            np.save(os.path.join(session_path, 'video_object.npy'), video)

            del video, list_of_fragments, list_of_global_fragments
