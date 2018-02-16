from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./')
sys.path.append('./utils')
sys.path.append('./network/identification_model')

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

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
    path_to_results_hard_drive = '/media/rhea/ground_truth_results_backup'
    tracked_videos_folder = os.path.join(path_to_results_hard_drive, 'tracked_videos')
    session_paths = [x[0] for x in os.walk(tracked_videos_folder) if 'session' in x[0][-16:] and 'Trash' not in x[0]]
    path_to_tracked_videos_data_frame = os.path.join(tracked_videos_folder, 'tracked_videos_data_frame.pkl')
    tracked_videos_data_frame = pd.read_pickle(path_to_tracked_videos_data_frame)

    plt.ion()
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    sns.set_style("ticks")
    fig, ax = plt.subplots(1,1)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None,
                wspace=None, hspace=None)
    velocities_BL = []
    for i, session_path in enumerate(session_paths):
        print("\nsession_path: ", session_path)
        species = tracked_videos_data_frame.loc[i].animal_type
        bad_video = tracked_videos_data_frame.loc[i].bad_video_example
        group_size = tracked_videos_data_frame.loc[i].number_of_animals
        if 'zebrafish' in species and not bad_video and 'nacre' not in species:
            color = 'g'
            plot_flag = True
        elif 'drosophila' in species and not bad_video:
            color = 'm'
            plot_flag = True
        elif 'drosophila (1)' in species and bad_video and group_size == 100:
            color = 'c'
            plot_flag = True
        elif 'drosophila (2)' in species and bad_video and group_size == 100:
            color = 'salmon'
            plot_flag = True
        elif 'drosophila' in species and bad_video and group_size == 60:
            color = 'y'
            plot_flag = True
        else:
            plot_flag = False

        if plot_flag:
            print("loading video")
            video = np.load(os.path.join(session_path, 'video_object.npy')).item()

            if not hasattr(video, 'velocities_BL') or not hasattr(video, 'minimum_number_of_frames_moving_in_first_global_fragment'):
            # if True:
                print("loading fragments")
                list_of_fragments = np.load(os.path.join(session_path, 'preprocessing', 'fragments.npy')).item()
                print("loading global_fragments")
                list_of_global_fragments = ListOfGlobalFragments.load(os.path.join(session_path, 'preprocessing', 'global_fragments.npy'),
                                                list_of_fragments.fragments)

                ### Overall distribution of velocities
                vels_good = np.hstack([f.frame_by_frame_velocity() for f in list_of_fragments.fragments if f.is_an_individual])
                vels_good = vels_good[vels_good != 0]
                body_length = video.median_body_length
                frame_rate = video.frames_per_second
                vels_good_BL = vels_good * frame_rate / body_length
                video.velocities_BL = vels_good_BL


                ### Distribution of frames in the fragments of the first global fragment
                moving_threshold = np.log10(0.75) # in log10(BL/s) so the threshold is 1BL/s
                list_of_global_fragments.order_by_distance_travelled()
                accumulation_trial = int(video.accumulation_folder[-1])
                global_fragment_index_used_for_accumulation_good = accumulation_trial if accumulation_trial == 0 else accumulation_trial - 1
                first_global_fragment_good = list_of_global_fragments.global_fragments[global_fragment_index_used_for_accumulation_good]
                frames_good, frames_moving_good = get_frames_and_frames_moving_for_fragments(first_global_fragment_good.individual_fragments, moving_threshold, frame_rate, body_length)
                video.minimum_number_of_frames_moving_in_first_global_fragment = 3 if np.min(frames_moving_good) < 3 else np.min(frames_moving_good)
                np.save(os.path.join(session_path, 'video_object.npy'), video)
                del list_of_fragments, list_of_global_fragments

            else:
                vels_good_BL = video.velocities_BL
            print(video.minimum_number_of_frames_moving_in_first_global_fragment)

            nbins = 100
            min = np.min(vels_good_BL)
            max = np.max(vels_good_BL)
            logbins = np.linspace(np.log10(min), np.log10(max), nbins)
            n, bins = np.histogram(np.log10(vels_good_BL), bins = logbins, normed = True)
            ax.plot(bins[:-1], n, linestyle = '-', color = color, linewidth = 3)
            ax.axvline(np.log10(0.75), color = 'r', linestyle = '--')
            ax.set_xticks([-2, -1, 0, 1])
            ax.set_xticklabels([0.01, 0.1, 1, 10])
            ax.set_xlabel('Speed [BL/s]', fontsize = 25)
            ax.set_ylabel('PDF', fontsize = 25)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_xlim((-2.5,1.5))
            ax.set_ylim((0,1.6))
            sns.despine(ax = ax, right = True, top = True)
            del video
            plt.show()
    fig.savefig(os.path.join(tracked_videos_folder,'velocities_distribution.pdf'), transparent = True)
