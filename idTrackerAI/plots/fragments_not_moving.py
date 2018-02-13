from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./')
sys.path.append('./utils')
sys.path.append('./network/identification_model')

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from fragment import Fragment
from list_of_fragments import ListOfFragments
from list_of_global_fragments import ListOfGlobalFragments
from GUI_utils import selectDir

def get_frames_and_frames_moving_for_fragments(fragments, moving_threshold, frame_rate, body_lengt):
    frames_moving = []
    frames = []
    for f in fragments:
        frames.append(f.number_of_images)
        velocities = f.frame_by_frame_velocity()
        velocities_BL = velocities * frame_rate / body_length
        frames_moving.append(sum(np.log10(velocities_BL) > moving_threshold))
    frames_moving = [f for f in frames_moving if f > 0]
    return frames, frames_moving

if __name__ == '__main__':
    path_to_results_hard_drive = '/media/atlas/ground_truth_results_backup'
    sessions = ['100 drosophila (females)/Canton_N100_11-23-17_12-59-17/session_20180122',
                    '10_flies_compressed_clara/session_20180207',
                    '38 drosophila (females males)/Canton_N38_top_video_01-31-18_10-50-14/session_20180201',
                    '60 drosophila (females)/Canton_N59_12-15-17_16-32-02/session_20180102',
                    '60 drosophila (females)/Canton_N60_12-15-17_15-15-10/session_20171221',
                    '72 drosophila (females - males)/session_20180201',
                    '80 drosophila (females - males)/session_20180206',
                    '80 drosophila (females males)/Canton_N80_11-28-17_17-21-32/session_20180123',
                    'idTrackerVideos/Moscas/2011dic12/Video_4fem_2mal_bottom/session_20180130',
                    'idTrackerVideos/Moscas/20121010/PlatoGrande_8females_2/session_20180131']
    path_to_results_hard_drive = '/media/chronos/ground_truth_results_backup'
    tracked_videos_folder = os.path.join(path_to_results_hard_drive, 'tracked_videos')
    session_paths = [x[0] for x in os.walk(tracked_videos_folder) if 'session' in x[0][-16:] and 'Trash' not in x[0]]

    plt.ion()
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    sns.set_style("ticks")

    for session, session_path in zip(sessions, session_paths):
        # session_path = os.path.join(path_to_results_hard_drive, 'tracked_videos', session)
        print("\nsession_path: ", session)
        # if 'drosophila' in session_path\
        #     or 'flies' in session_path\
        #     or ('Moscas' in session_path and 'PlatoGrande' in session_path)\
        #     or ('Moscas' in session_path and 'Platogrande' in session_path):
        if True:

            print("loading video")
            video = np.load(os.path.join(session_path, 'video_object.npy')).item()
            body_length = video.median_body_length
            frame_rate = video.frames_per_second
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

            vels_good_BL = vels_good * frame_rate / body_length
            min = np.min(vels_good_BL)
            max = np.max(vels_good_BL)
            logbins = np.linspace(np.log10(min), np.log10(max), nbins)

            plt.ion()
            fig, ax_arr = plt.subplots(1,2)
            fig.set_size_inches((screen_x/100,screen_y/100))
            fig.suptitle(session)
            ax_arr[0].hist(np.log10(vels_good_BL), bins = logbins, normed = True)
            ax_arr[0].set_xticks([-1, 0, 1])
            ax_arr[0].set_xticklabels([0.1, 1, 10])
            ax_arr[0].set_xlabel('BL/s')
            ax_arr[1].hist(np.log10(vels_good), bins = logbins, normed = True)
            ax_arr[1].set_xticks([-1, 0, 1])
            ax_arr[1].set_xticklabels([0.1, 1, 10])
            ax_arr[1].set_xlabel('px/frame')
            moving_threshold = 0 # in log10(BL/s) so the threshold is 1BL/s

            ### Distribution of frames in the fragments of the first global fragment
            list_of_global_fragments.order_by_distance_travelled()
            accumulation_trial = int(video.accumulation_folder[-1])
            global_fragment_index_used_for_accumulation_good = accumulation_trial if accumulation_trial == 0 else accumulation_trial - 1
            first_global_fragment_good = list_of_global_fragments.global_fragments[global_fragment_index_used_for_accumulation_good]
            frames_good, frames_moving_good = get_frames_and_frames_moving_for_fragments(first_global_fragment_good.individual_fragments, moving_threshold, frame_rate, body_length)
            video.minimum_number_of_frames_moving_in_first_global_fragment = 3 if np.min(frames_moving_good) < 3 else np.min(frames_moving_good)
            print(video.minimum_number_of_frames_moving_in_first_global_fragment)

            np.save(os.path.join(session_path, 'video_object.npy'), video)
            fig.savefig(os.path.join(session_path,'velocities_distribution.pdf'), transparent = True)

            del video, list_of_fragments, list_of_global_fragments
