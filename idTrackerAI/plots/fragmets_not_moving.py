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
    session_paths = ['/home/prometheus/Desktop/IdTrackerDeep/videos/60 drosophila (females)/Canton_N60_12-15-17_15-15-10/session_20180110']

    for session_path in session_paths:
        print("\nsession_path: ", session_path)
        if 'drosophila' in session_path\
            or 'flies' in session_path\
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
            print(video.minimum_number_of_frames_moving_in_first_global_fragment)

            np.save(os.path.join(session_path, 'video_object.npy'), video)

            # del video, list_of_fragments, list_of_global_fragments
