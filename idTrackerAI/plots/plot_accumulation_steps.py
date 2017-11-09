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

def plot_accuracy_step(ax, accumulation_step, training_dict, validation_dict):
    total_epochs_completed_previous_step = int(np.sum(training_dict['number_of_epochs_completed'][:accumulation_step]))
    total_epochs_completed_this_step = int(np.sum(training_dict['number_of_epochs_completed'][:accumulation_step + 1]))

    ax.plot(training_dict['accuracy'][0:total_epochs_completed_previous_step], '-r', alpha = .3)
    ax.plot(validation_dict['accuracy'][0:total_epochs_completed_previous_step], '-b', alpha = .3)

    ax.plot(range(total_epochs_completed_previous_step, total_epochs_completed_this_step),
                training_dict['accuracy'][total_epochs_completed_previous_step:total_epochs_completed_this_step], '-r', label = 'training')
    ax.plot(range(total_epochs_completed_previous_step, total_epochs_completed_this_step),
                validation_dict['accuracy'][total_epochs_completed_previous_step:total_epochs_completed_this_step], '-b', label = 'validation')


def plot_validation_and_training_accuracies(video, ax):
    validation_individual_accuracies = getattr(video, 'validation_individual_accuracies_' + str(video.accumulation_trial))[-1]
    training_individual_accuracies = getattr(video, 'training_individual_accuracies_' + str(video.accumulation_trial))[-1]

    width = 0.35
    ax.bar(np.arange(video.number_of_animals)+1, validation_individual_accuracies, width, color = 'b', alpha=.5, label = 'validation', align = 'edge')
    ax.bar(np.arange(video.number_of_animals)+1-width, training_individual_accuracies, width, color = 'r', alpha=.5, label = 'training', align = 'edge')

def plot_individual_certainty(video, ax, colors):
    ax.bar(np.arange(video.number_of_animals)+1, video.individual_P2, color = colors[1:])
    ax.axhline(1., color = 'k', linestyle = '--')

def set_properties_fragments(video, fig, ax_arr, number_of_accumulation_steps):

    [ax.set_ylabel('Individual', fontsize = 12) for ax in ax_arr]
    [ax.set_yticks(range(0,video.number_of_animals,2)) for ax in ax_arr]
    [ax.set_yticklabels(range(1,video.number_of_animals + 1,2)) for ax in ax_arr]
    [ax.set_xticklabels([]) for ax in ax_arr[:-1]]
    [ax.set_xlim([0., video.number_of_frames]) for ax in ax_arr]
    [ax.set_ylim([-.5, video.number_of_animals + .5 - 1]) for ax in ax_arr]
    for i, ax in enumerate(ax_arr):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + .015 * i , pos.width, pos.height])

    ax_arr[-1].set_xlabel('Frame number', fontsize = 12)

def set_properties_network_accuracy(video, fig, ax_arr, number_of_accumulation_steps, total_number_of_epochs_completed):

    [ax.set_ylabel('Accuracy', fontsize = 12) for ax in ax_arr]
    [ax.set_xticklabels([]) for ax in ax_arr[:-1]]
    [ax.set_xlim([0., total_number_of_epochs_completed]) for ax in ax_arr]
    [ax.set_ylim([0, 1.05]) for ax in ax_arr]
    for i, ax in enumerate(ax_arr):
        pos = ax.get_position()
        ax.set_position([pos.x0 + .05, pos.y0 + .015 * i , pos.width - 0.05, pos.height])

    ax_arr[-1].set_xlabel('Epoch number', fontsize = 12)
    ax_arr[0].legend()

    axes_position_first_acc = ax_arr[0].get_position()
    axes_position_last_acc = ax_arr[number_of_accumulation_steps - 1].get_position()
    text_axes = fig.add_axes([axes_position_last_acc.x0 + axes_position_last_acc.width + .01, axes_position_last_acc.y0, 0.01, (axes_position_first_acc.y0 + axes_position_first_acc.height) - axes_position_last_acc.y0])
    text_axes.text(0.5, 0.5,'Accumulation steps', horizontalalignment='center', verticalalignment='center', rotation=-90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)


def set_properties_assignment(video, fig, ax, number_of_accumulation_steps):
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + .008 * number_of_accumulation_steps , pos.width, pos.height])

    axes_position_assignment = ax.get_position()
    text_axes = fig.add_axes([axes_position_assignment.x0 + axes_position_assignment.width + .01, axes_position_assignment.y0, 0.01, axes_position_assignment.height])
    text_axes.text(0.5, 0.5,'Assignment', horizontalalignment='center', verticalalignment='center', rotation=-90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)

    ax.set_yticks(range(0,video.number_of_animals,2))
    ax.set_yticklabels(range(1,video.number_of_animals + 1,2))
    ax.set_xlim([0., video.number_of_frames])
    ax.set_ylim([-.5, video.number_of_animals + .5 - 1])
    ax.set_xlabel('Frame number', fontsize = 12)
    ax.set_ylabel('Individual', fontsize = 12)

def set_properties_final_accuracy(video, fig, ax, number_of_accumulation_steps):
    ax.set_xlabel('Individual', fontsize = 12)
    ax.set_ylabel('Average \ncertainty', fontsize = 12)

def get_list_of_accuulation_steps_to_plot(number_of_accumulation_steps):

    if number_of_accumulation_steps <= 4:
        return range(number_of_accumulation_steps)
    elif number_of_accumulation_steps > 4:
        list_of_accumulation_steps = [0, 1]
        intermidiate_accumulation_step = int(np.ceil((number_of_accumulation_steps-2)/2) + 2)
        list_of_accumulation_steps.append(intermidiate_accumulation_step)
        list_of_accumulation_steps.append(number_of_accumulation_steps-1)
        return list_of_accumulation_steps

def plot_accumulation_steps(video, list_of_fragments, training_dict, validation_dict):
    plt.ion()
    sns.set_style("ticks")
    number_of_accumulation_steps = get_number_of_accumulation_steps(list_of_fragments)

    list_of_accumulation_steps = get_list_of_accuulation_steps_to_plot(number_of_accumulation_steps)

    fig1 = plt.figure()
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig1.set_size_inches((screen_x*2/3/100,screen_y/100))
    colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=True)

    ax_arr_fragments = []
    ax_arr_network_accuracy = []
    print("list_of_accumulation_steps ", list_of_accumulation_steps)
    number_of_accumulation_steps_to_plot = len(list_of_accumulation_steps)
    for i, accumulation_step in enumerate(list_of_accumulation_steps):
        ax = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (i, 0), colspan=3)
        ax_arr_fragments.append(ax)
        plot_accumulation_step_from_fragments(list_of_fragments.fragments, ax, accumulation_step, False, colors)

        ax = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (i, 3), colspan=2)
        ax_arr_network_accuracy.append(ax)
        plot_accuracy_step(ax, accumulation_step, training_dict, validation_dict)

    ax_assignment = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (i + 1, 0), colspan=5)
    plot_accumulation_step_from_fragments(list_of_fragments.fragments, ax_assignment, accumulation_step, True, colors)

    ax_final_accuracy = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (i + 2, 0), colspan=5)
    plot_individual_certainty(video, ax_final_accuracy, colors)

    fig1.subplots_adjust(left=None, bottom=.1, right=None, top=.95,
                wspace=None, hspace=.5)
    set_properties_fragments(video, fig1, ax_arr_fragments, number_of_accumulation_steps_to_plot)
    set_properties_network_accuracy(video, fig1, ax_arr_network_accuracy, number_of_accumulation_steps_to_plot, np.sum(training_dict['number_of_epochs_completed']))
    set_properties_assignment(video, fig1, ax_assignment, number_of_accumulation_steps_to_plot)
    set_properties_final_accuracy(video, fig1, ax_final_accuracy, number_of_accumulation_steps_to_plot)

    fig1.savefig(os.path.join(video._accumulation_folder,'accumulation_steps.pdf'), transparent=False)
    plt.show()

if __name__ == '__main__':
    # session_path = selectDir('./') #select path to video
    session_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4'
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item(0)
    list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_fragments_dictionaries = np.load(os.path.join(video._accumulation_folder,'light_list_of_fragments.npy'))
    fragments = [Fragment(number_of_animals = video.number_of_animals) for fragment_dictionary in list_of_fragments_dictionaries]
    [fragment.__dict__.update(fragment_dictionary) for fragment, fragment_dictionary in zip(fragments, list_of_fragments_dictionaries)]
    light_list_of_fragments = ListOfFragments(video, fragments)
    training_dict = np.load(os.path.join(video.accumulation_folder, 'training_loss_acc_dict.npy')).item()
    validation_dict = np.load(os.path.join(video.accumulation_folder, 'validation_loss_acc_dict.npy')).item()
    plot_accumulation_steps(video, light_list_of_fragments, training_dict, validation_dict)
