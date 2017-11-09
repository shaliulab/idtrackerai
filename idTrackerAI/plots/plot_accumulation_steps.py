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


LABELS = ['crossing', 'assigned', 'not assigned']
COLORS = ['k', 'g', 'r']

def get_object_type(object_to_evaluate):

    if object_to_evaluate.is_a_crossing:
        return 0 #crossing
    elif object_to_evaluate.assigned_identity != 0:
        return 1 #assigned after accumulation
    else:
        return 2 #not assigned

def get_number_of_accumulation_steps(list_of_fragments):
    return len(np.unique([fragment.accumulation_step for fragment in list_of_fragments.fragments
                        if fragment.accumulation_step is not None]))

def plot_accumulation_step_from_fragments(fragments, ax, accumulation_step, plot_assignment_flag, colors):
    for fragment in fragments:
        if fragment.is_an_individual \
            and (fragment.used_for_training or plot_assignment_flag and not fragment.used_for_training) \
            and fragment.accumulation_step <= accumulation_step:
            blob_index = fragment.assigned_identity-1
            (start, end) = fragment.start_end
            ax.add_patch(
                patches.Rectangle(
                    (start, blob_index - 0.5),   # (x,y)
                    end - start - 1,  # width
                    1.,          # height
                    fill=True,
                    edgecolor=None,
                    facecolor=colors[fragment.assigned_identity],
                    alpha = 1. if fragment.used_for_training else .5
                )
            )

def plot_validation_and_training_accuracies(video, ax):
    validation_individual_accuracies = getattr(video, 'validation_individual_accuracies_' + str(video.accumulation_trial))[-1]
    training_individual_accuracies = getattr(video, 'training_individual_accuracies_' + str(video.accumulation_trial))[-1]

    width = 0.35
    ax.bar(np.arange(video.number_of_animals)+1-width, training_individual_accuracies, width, color = 'r', alpha=.5, label = 'training', align = 'edge')
    ax.bar(np.arange(video.number_of_animals)+1, validation_individual_accuracies, width, color = 'b', alpha=.5, label = 'validation', align = 'edge')

def plot_individual_certainty(video, ax, colors):
    ax.bar(np.arange(video.number_of_animals)+1, video.individual_P2, color = colors[1:])
    ax.axhline(1., color = 'k', linestyle = '--')

def plot_accumulation_steps(video, list_of_fragments):
    plt.ion()
    sns.set_style("ticks")
    number_of_accumulation_steps = get_number_of_accumulation_steps(list_of_fragments)


    fig1, ax_arr = plt.subplots(number_of_accumulation_steps + 2, 1)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig1.set_size_inches((screen_x*2/3/100,screen_y/100))
    colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=True)

    for accumulation_step in range(number_of_accumulation_steps):
        ax = ax_arr[accumulation_step]
        plot_accumulation_step_from_fragments(list_of_fragments.fragments, ax, accumulation_step, False, colors)


    plot_accumulation_step_from_fragments(list_of_fragments.fragments, ax_arr[-2], accumulation_step, True, colors)

    plot_individual_certainty(video, ax_arr[-1], colors)

    set_properties_fig1(video, fig1, ax_arr, number_of_accumulation_steps)

    fig1.savefig(os.path.join(video._accumulation_folder,'accumulation_steps.pdf'), transparent=False)
    plt.show()

def set_properties_fig1(video, fig, ax_arr, number_of_accumulation_steps):

    fig.subplots_adjust(left=None, bottom=.1, right=None, top=.95,
                wspace=None, hspace=.5)

    [ax.set_ylabel('Individual', fontsize = 12) for ax in ax_arr[:-2]]
    [ax.set_yticks(range(0,video.number_of_animals,2)) for ax in ax_arr[:-2]]
    [ax.set_yticklabels(range(1,video.number_of_animals + 1,2)) for ax in ax_arr[:-2]]
    [ax.set_xticklabels([]) for ax in ax_arr[:-2]]
    [ax.set_xlim([0., video.number_of_frames]) for ax in ax_arr[:-2]]
    [ax.set_ylim([-.5, video.number_of_animals + .5 - 1]) for ax in ax_arr[:-2]]
    for i, ax in enumerate(ax_arr[:-1]):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + .015 * i , pos.width, pos.height])

    axes_position_first_acc = ax_arr[0].get_position()
    axes_position_last_acc = ax_arr[number_of_accumulation_steps - 1].get_position()
    text_axes = fig.add_axes([axes_position_last_acc.x0 + axes_position_last_acc.width + .01, axes_position_last_acc.y0, 0.01, (axes_position_first_acc.y0 + axes_position_first_acc.height) - axes_position_last_acc.y0])
    text_axes.text(0.5, 0.5,'Accumulation steps', horizontalalignment='center', verticalalignment='center', rotation=-90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)

    axes_position_assignment = ax_arr[-2].get_position()
    text_axes = fig.add_axes([axes_position_assignment.x0 + axes_position_assignment.width + .01, axes_position_assignment.y0, 0.01, axes_position_assignment.height])
    text_axes.text(0.5, 0.5,'Assignment', horizontalalignment='center', verticalalignment='center', rotation=-90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)

    ax_arr[-2].set_yticks(range(0,video.number_of_animals,2))
    ax_arr[-2].set_yticklabels(range(1,video.number_of_animals + 1,2))
    ax_arr[-2].set_xlim([0., video.number_of_frames])
    ax_arr[-2].set_ylim([-.5, video.number_of_animals + .5 - 1])
    ax_arr[-2].set_xlabel('Frame number', fontsize = 15)
    ax_arr[-2].set_ylabel('Individual', fontsize = 12)

    ax_arr[-1].set_xlabel('Individual', fontsize = 15)
    ax_arr[-1].set_ylabel('Average \ncertainty', fontsize = 12)

if __name__ == '__main__':
    # session_path = selectDir('./') #select path to video
    session_path = '/home/themis/Desktop/IdTrackerDeep/videos/idTrackerDeep_LargeGroups_3/100fish/First/session_2'
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item(0)
    # list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_fragments_dictionaries = np.load(os.path.join(video._accumulation_folder,'light_list_of_fragments.npy'))
    fragments = [Fragment(number_of_animals = video.number_of_animals) for fragment_dictionary in list_of_fragments_dictionaries]
    [fragment.__dict__.update(fragment_dictionary) for fragment, fragment_dictionary in zip(fragments, list_of_fragments_dictionaries)]
    light_list_of_fragments = ListOfFragments(video, fragments)
    plot_accumulation_steps(video, light_list_of_fragments)
