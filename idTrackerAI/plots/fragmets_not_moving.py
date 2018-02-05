import sys
sys.path.append('./')
sys.path.append('./network/identification_model')

import numpy as np
from matplotlib import pyplot as plt

from fragment import Fragment
from list_of_fragments import ListOfFragments
from list_of_global_fragments import ListOfGlobalFragments

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
    video_good = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/test_flies/video_object.npy').item()
    list_of_fragments_good = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/test_flies/fragments.npy').item()
    list_of_global_fragments_good = ListOfGlobalFragments.load('/home/chronos/Desktop/IdTrackerDeep/videos/test_flies/global_fragments.npy', list_of_fragments_good.fragments)
    video_bad = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/test_flies/video_object_2.npy').item()
    list_of_fragments_bad = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/test_flies/fragments_2.npy').item()
    list_of_global_fragments_bad = ListOfGlobalFragments.load('/home/chronos/Desktop/IdTrackerDeep/videos/test_flies/global_fragments_2.npy', list_of_fragments_bad.fragments)

    ### Overall distribution of velocities
    nbins = 100
    vels_good = np.hstack([f.frame_by_frame_velocity() for f in list_of_fragments_good.fragments if f.is_an_individual])
    vels_bad = np.hstack([f.frame_by_frame_velocity() for f in list_of_fragments_bad.fragments if f.is_an_individual])
    vels_good = vels_good[vels_good != 0]
    vels_bad = vels_bad[vels_bad != 0]
    min = np.min(np.hstack([vels_good, vels_bad]))
    max = np.max(np.hstack([vels_good, vels_bad]))
    logbins = np.linspace(np.log10(min), np.log10(max), nbins)

    plt.ion()
    fig, ax_arr = plt.subplots(1,2)
    ax_arr[0].set_title('good')
    ax_arr[0].hist(np.log10(vels_good), bins = logbins, normed = True)
    ax_arr[1].set_title('bad')
    ax_arr[1].hist(np.log10(vels_bad), bins = logbins, normed = True)
    moving_threshold = .17

    ### Compute number of frames number_of_frames
    frames_good, frames_moving_good = get_frames_and_frames_moving_for_fragments(list_of_fragments_good.fragments)
    frames_bad, frames_moving_bad = get_frames_and_frames_moving_for_fragments(list_of_fragments_bad.fragments)

    nbins = 25
    min = 1
    max = np.max(frames_good + frames_bad)
    logbins = np.linspace(np.log10(min), np.log10(max), nbins)

    fig, ax_arr = plt.subplots(2,2, sharex = True, sharey = True)
    ax_arr[0,0].set_title('good')
    ax_arr[0,0].hist(np.log10(frames_good), bins = logbins, normed = True)
    ax_arr[1,0].hist(np.log10(frames_moving_good), bins = logbins, normed = True)
    ax_arr[0,1].set_title('bad')
    ax_arr[0,1].hist(np.log10(frames_bad), bins = logbins, normed = True)
    ax_arr[1,1].hist(np.log10(frames_moving_bad), bins = logbins, normed = True)

    ### Distribution of frames in the fragments of the first global fragment
    list_of_global_fragments_good.order_by_distance_travelled()
    accumulation_trial = int(video_good.accumulation_folder[-1])
    global_fragment_index_used_for_accumulation_good = accumulation_trial if accumulation_trial == 0 else accumulation_trial - 1
    print(global_fragment_index_used_for_accumulation_good)
    first_global_fragment_good = list_of_global_fragments_good.global_fragments[global_fragment_index_used_for_accumulation_good]

    list_of_global_fragments_bad.order_by_distance_travelled()
    accumulation_trial = int(video_bad.accumulation_folder[-1])
    global_fragment_index_used_for_accumulation_bad = accumulation_trial if accumulation_trial == 0 else accumulation_trial - 1
    print(global_fragment_index_used_for_accumulation_bad)
    first_global_fragment_bad = list_of_global_fragments_bad.global_fragments[global_fragment_index_used_for_accumulation_bad]
    video_bad

    frames_good, frames_moving_good = get_frames_and_frames_moving_for_fragments(first_global_fragment_good.individual_fragments)
    frames_bad, frames_moving_bad = get_frames_and_frames_moving_for_fragments(first_global_fragment_bad.individual_fragments)

    nbins = 25
    min = 1
    max = np.max(frames_good + frames_bad)
    logbins = np.linspace(np.log10(min), np.log10(max), nbins)

    fig, ax_arr = plt.subplots(2,2, sharex = True, sharey = True)
    ax_arr[0,0].set_title('good')
    ax_arr[0,0].hist(np.log10(frames_good), bins = logbins, normed = True)
    ax_arr[1,0].hist(np.log10(frames_moving_good), bins = logbins, normed = True)
    ax_arr[0,1].set_title('bad')
    ax_arr[0,1].hist(np.log10(frames_bad), bins = logbins, normed = True)
    ax_arr[1,1].hist(np.log10(frames_moving_bad), bins = logbins, normed = True)
