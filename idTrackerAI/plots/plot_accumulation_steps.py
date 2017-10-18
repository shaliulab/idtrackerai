from __future__ import absolute_import, division, print_function
import os
import sys
from glob import glob
sys.path.append('./')
sys.path.append('./utils')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns

from video import Video
from list_of_blobs import ListOfBlobs
from blob import Blob
from list_of_fragments import ListOfFragments
from fragment import Fragment
# from GUI_utils import selectDir
from py_utils import get_spaced_colors_util

LABELS = ['crossings', 'assigned during accumulation', 'assigned after accumulation', 'not assigned']
COLORS = ['k', 'g', 'y', 'r']

def get_object_type(object_to_evaluate):

    if object_to_evaluate.is_a_crossing:
        return 0 #crossing
    elif object_to_evaluate.is_an_individual and object_to_evaluate.used_for_training:
        return 1 #assigned during accumulation
    elif object_to_evaluate.final_identity != 0:
        return 2 #assigned after accumulation
    else:
        return 3 #not assigned

def get_number_of_accumulation_steps(list_of_fragments):
    return len(np.unique([fragment.accumulation_step for fragment in list_of_fragments.fragments
                        if fragment.accumulation_step is not None]))

def plot_accumulation_step_from_blobs(video, blobs_in_video, ax, accumulation_step, plot_assignment_flag):

    def get_number_of_blobs_per_class_in_frame(blobs_in_frame, number_of_animals, accumulation_step, plot_assignment_flag):
        number_of_blobs_per_class = np.zeros(len(LABELS))
        number_of_individuals_in_frame = sum([blob.is_an_individual for blob in blobs_in_frame])
        for blob in blobs_in_frame:
            if (blob.accumulation_step is not None and blob.accumulation_step <= accumulation_step) or plot_assignment_flag:
                number_of_blobs_per_class[get_object_type(blob)] += 1
        number_of_blobs_per_class[0] = number_of_animals - number_of_individuals_in_frame
        return number_of_blobs_per_class

    def plot_bar_of_classes(ax, frame_number, number_of_blobs_per_class, plot_assignment_flag):
        bottom = 0
        for _type, color in enumerate(COLORS):
            if _type <= 1 or (plot_assignment_flag and _type >= 2):
                ax.bar(frame_number, number_of_blobs_per_class[_type], color = color, align = 'center', bottom = bottom, width = 1.)
                bottom += number_of_blobs_per_class[_type]

    for frame_number, blobs_in_frame in enumerate(blobs_in_video):
        number_of_blobs_per_class = get_number_of_blobs_per_class_in_frame(blobs_in_frame, video.number_of_animals, accumulation_step, plot_assignment_flag)
        plot_bar_of_classes(ax, frame_number, number_of_blobs_per_class, plot_assignment_flag)


def plot_accumulation_step_from_fragments(fragments, ax, accumulation_step, plot_assignment_flag, colors):
    for fragment in fragments:
        _type = get_object_type(fragment)
        if (_type == 1 or plot_assignment_flag and _type >= 2) and fragment.accumulation_step <= accumulation_step:
            blob_index = fragment.blob_hierarchy_in_starting_frame
            (start, end) = fragment.start_end
            ax.add_patch(
                patches.Rectangle(
                    (start, blob_index - 0.5),   # (x,y)
                    end - start - 1,  # width
                    1.,          # height
                    fill=True,
                    edgecolor=None,
                    facecolor=colors[fragment.final_identity],
                    alpha = 1.
                )
            )

def plot_accumulation_steps(video, list_of_fragments, list_of_blobs):
    plt.ion()
    sns.set_style("ticks")
    number_of_accumulation_steps = get_number_of_accumulation_steps(list_of_fragments)

    fig1, ax_arr1 = plt.subplots(number_of_accumulation_steps+1, 1)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig1.set_size_inches((screen_x*2/3/100,screen_y/100))
    fig2, ax_arr2 = plt.subplots(number_of_accumulation_steps+1, 1)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig2.set_size_inches((screen_x*2/3/100,screen_y/100))

    colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=True)

    for accumulation_step in range(number_of_accumulation_steps + 1):
        plot_assignment_flag = accumulation_step == number_of_accumulation_steps
        ax1 = ax_arr1[accumulation_step]
        ax2 = ax_arr2[accumulation_step]
        plot_accumulation_step_from_fragments(list_of_fragments.fragments, ax1, accumulation_step, plot_assignment_flag, colors)
        plot_accumulation_step_from_blobs(video, list_of_blobs.blobs_in_video, ax2, accumulation_step, plot_assignment_flag)

    set_properties_fig1(video, ax_arr1)
    set_properties_fig2(video, ax_arr2)

    fig1.savefig(os.path.join(video._accumulation_folder,'accumulation_steps_1.png'), transparent=False)
    fig2.savefig(os.path.join(video._accumulation_folder,'accumulation_steps_2.png'), transparent=False)
    plt.show()

def set_properties_fig1(video, ax_arr):
    ax_arr[-1].set_xlabel('Frame number')
    [ax.set_ylabel('Blob index') for ax in ax_arr]
    [ax.set_yticks(range(0,video.number_of_animals,2)) for ax in ax_arr]
    [ax.set_yticklabels(range(1,video.number_of_animals + 1,2)) for ax in ax_arr]
    [ax.set_xticklabels([]) for ax in ax_arr[:-1]]
    [ax.set_xlim([0., video.number_of_frames]) for ax in ax_arr]
    [ax.set_ylim([-.5, video.number_of_animals + .5 - 1]) for ax in ax_arr]

def set_properties_fig2(video, ax_arr):
    l_crossings = mpatches.Patch(color=COLORS[0], alpha=1., linewidth=0)
    l_accumulated = mpatches.Patch(color=COLORS[1], alpha=1., linewidth=0)
    l_assigned = mpatches.Patch(color=COLORS[2], alpha=1., linewidth=0)
    l_not_assigned = mpatches.Patch(color=COLORS[3], alpha=1., linewidth=0)
    ax_arr[0].legend((l_crossings, l_accumulated, l_assigned, l_not_assigned),
                ('Crossing or occlusion', 'Assigned during accumulation', 'Assigned after accumulation', 'Not assigned'))
    ax_arr[-1].set_xlabel('Frame number')
    [ax.set_ylabel('N. of idniv') for ax in ax_arr]
    [ax.set_xticklabels([]) for ax in ax_arr[:-1]]
    [ax.set_xlim([0., video.number_of_frames]) for ax in ax_arr]
    [ax.set_ylim([0, video.number_of_animals]) for ax in ax_arr]

if __name__ == '__main__':
    # session_path = selectDir('./') #select path to video
    session_path = '/home/chronos/Desktop/IdTrackerDeep/videos/conflicto_short/session_23'
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item(0)
    list_of_blobs = ListOfBlobs.load(video.blobs_path)
    list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_fragments_dictionaries = np.load(os.path.join(video._accumulation_folder,'light_list_of_fragments.npy'))
    fragments = [Fragment(number_of_animals = video.number_of_animals) for fragment_dictionary in list_of_fragments_dictionaries]
    [fragment.__dict__.update(fragment_dictionary) for fragment, fragment_dictionary in zip(fragments, list_of_fragments_dictionaries)]
    light_list_of_fragments = ListOfFragments(video, fragments)
    plot_accumulation_steps(video, light_list_of_fragments, list_of_blobs)
