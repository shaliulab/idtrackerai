from __future__ import absolute_import, division, print_function
import os
import sys
from glob import glob
sys.path.append('./')
sys.path.append('./utils')
sys.path.append('./groundtruth_utils')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
from pprint import pprint
import collections

from video import Video
from list_of_blobs import ListOfBlobs
from blob import Blob
from list_of_fragments import ListOfFragments
from fragment import Fragment
from GUI_utils import selectDir
from py_utils import get_spaced_colors_util
from generate_groundtruth import GroundTruthBlob, GroundTruth

LABELS = ['crossing', 'assigned', 'not assigned']
COLORS = ['k', 'g', 'r']

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

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

def plot_accumulation_step_from_fragments(fragments, ax, accumulation_step, plot_assignment_flag, colors, identity_to_blob_hierarchy_list):
    for fragment in fragments:
        if fragment.is_an_individual:
            # blob_index = fragment.assigned_identity-1
            blob_index = identity_to_blob_hierarchy_list[fragment.final_identity-1]
            # blob_index = fragment.blob_hierarchy_in_starting_frame
            (start, end) = fragment.start_end
            if fragment.used_for_training and fragment.accumulation_step <= accumulation_step:
                color = colors[fragment.assigned_identity-1]
                alpha = 1.
            elif not plot_assignment_flag:
                color = 'w'
                alpha = 1.
            elif plot_assignment_flag:
                ax.add_patch(patches.Rectangle(
                        (start, blob_index - 0.5),   # (x,y)
                        end - start - 1,  # width
                        1.,          # height
                        fill = True,
                        edgecolor = None,
                        facecolor = 'w'))
                if fragment.assigned_identity != 0:
                    color = colors[fragment.assigned_identity-1]
                    alpha = .5
                else:
                    color = 'w'
                    alpha = 1.

            ax.add_patch(patches.Rectangle(
                    (start, blob_index - 0.5),   # (x,y)
                    end - start - 1,  # width
                    1.,          # height
                    fill = True,
                    edgecolor = None,
                    facecolor = color,
                    alpha = alpha))

def plot_crossings_step(no_gaps_fragments, ax, colors):
    for identity in no_gaps_fragments:
        for fragment in no_gaps_fragments[identity]:
            blob_index = identity - 1
            (start, end) = fragment
            color = colors[identity-1]
            alpha = 1.
            ax.add_patch(patches.Rectangle(
                    (start, blob_index - 0.5),   # (x,y)
                    end - start - 1,  # width
                    1.,          # height
                    fill = True,
                    edgecolor = None,
                    facecolor = color,
                    alpha = alpha))

def plot_accuracy_step(ax, accumulation_step, training_dict, validation_dict):
    total_epochs_completed_previous_step = int(np.sum(training_dict['number_of_epochs_completed'][:accumulation_step]))
    total_epochs_completed_this_step = int(np.sum(training_dict['number_of_epochs_completed'][:accumulation_step + 1]))

    ax.plot(np.asarray(training_dict['accuracy'][0:total_epochs_completed_previous_step]), '-r', alpha = .3)
    ax.plot(np.asarray(validation_dict['accuracy'][0:total_epochs_completed_previous_step]), '-b', alpha = .3)

    ax.plot(range(total_epochs_completed_previous_step, total_epochs_completed_this_step),
                np.asarray(training_dict['accuracy'][total_epochs_completed_previous_step:total_epochs_completed_this_step]), '-r', label = 'training')
    ax.plot(range(total_epochs_completed_previous_step, total_epochs_completed_this_step),
                np.asarray(validation_dict['accuracy'][total_epochs_completed_previous_step:total_epochs_completed_this_step]), '-b', label = 'validation')

def plot_individual_certainty_and_accuracy(video, ax_P2, ax_gt_accuracy, colors, identity_to_blob_hierarchy_list):
    # ax.bar(np.arange(video.number_of_animals)+1, video.individual_P2, color = colors[1:])
    n_bins = 20
    accuracy = np.asarray(video.gt_accuracy_interpolated['individual_accuracy'].values()) * 100
    individual_P2 = np.asarray(video.gt_accuracy_interpolated['individual_P2_in_validated_part'].values()) * 100
    minimum = np.min(np.hstack((accuracy,individual_P2)))
    maximum = 100
    n, bins, _ = ax_P2.hist(individual_P2, bins = np.linspace(minimum - 0.0005, maximum, n_bins), color = '.5', alpha = .5)
    bin_width = np.diff(bins)[0]
    for height, bin in zip(n,bins[:-1]):
        if height != 0:
            ax_P2.text(bin + bin_width/2, height + 0.1, str(int(height)), fontsize = 10, horizontalalignment='center')
    P2_mean = video.gt_accuracy_interpolated['mean_individual_P2_in_validated_part'] * 100
    P2_std = np.std(individual_P2)
    ax_P2.text(bins[0], ax_P2.get_ylim()[1]*.8, 'mean = %.2f ' %P2_mean + r'%', fontsize = 12)

    n_accuracy, bins_accuracy, _ = ax_gt_accuracy.hist(accuracy, bins = np.linspace(minimum - 0.0005, maximum, n_bins), color = 'g', alpha = .5)
    for height, bin in zip(n_accuracy,bins_accuracy[:-1]):
        if height != 0:
            ax_gt_accuracy.text(bin + bin_width/2, height + 0.1, str(int(height)), fontsize = 10, horizontalalignment='center')
    accuracy_mean = video.gt_accuracy_interpolated['accuracy'] * 100
    accuracy_std = np.std(accuracy)
    ax_gt_accuracy.text(bins[0], ax_gt_accuracy.get_ylim()[1]*.8, 'mean = %.3f' %accuracy_mean + r'%', fontsize = 12)

def set_properties_fragments(video, fig, ax_arr, number_of_accumulation_steps, list_of_accumulation_steps, zoomed_frames):

    [ax.set_ylabel('Step %i \n\nIndividual' %i, fontsize = 12) for ax, i in zip(ax_arr, np.asarray(list_of_accumulation_steps) + 1)]
    [ax.set_yticks(range(19,video.number_of_animals+1,20)) for ax in ax_arr]
    [ax.set_yticklabels(range(20,video.number_of_animals+1,20)) for ax in ax_arr]
    [ax.set_xticklabels([]) for ax in ax_arr[:-1]]
    [ax.set_xlim([0., video.number_of_frames]) for ax in ax_arr]
    [ax.set_ylim([-.5, video.number_of_animals + .5 - 1]) for ax in ax_arr]
    for i, ax in enumerate(ax_arr):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + .01 * i , pos.width, pos.height])
    [sns.despine(ax = ax, left=False, bottom=False, right=True) for ax in ax_arr]
    ax_arr[-1].add_patch(patches.Rectangle(
            (zoomed_frames[0], 0 - 0.5),   # (x,y)
            zoomed_frames[1] - zoomed_frames[0] + 1,  # width
            video.number_of_animals,          # height
            fill = False,
            edgecolor = 'k',
            linewidth = 2.,
            linestyle = '--',
            facecolor = None))
    ax_arr[-1].set_xlabel('Frame number', fontsize = 12)


    axes_position_first_acc = ax_arr[0].get_position()
    axes_position_last_acc = ax_arr[number_of_accumulation_steps - 1].get_position()
    text_axes = fig.add_axes([axes_position_last_acc.x0 - .1, axes_position_last_acc.y0, 0.01, (axes_position_first_acc.y0 + axes_position_first_acc.height) - axes_position_last_acc.y0])
    text_axes.text(0.5, 0.5,'Training and identification', horizontalalignment='center', verticalalignment='center', rotation=90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)

def set_properties_network_accuracy(video, fig, ax_arr, number_of_accumulation_steps, total_number_of_epochs_completed):

    [ax.set_ylabel('Accuracy', fontsize = 12) for ax in ax_arr]
    [ax.set_xticklabels([]) for ax in ax_arr[:-1]]
    [ax.set_xlim([0., total_number_of_epochs_completed]) for ax in ax_arr]
    [ax.set_ylim([0, 1.01]) for ax in ax_arr]
    for i, ax in enumerate(ax_arr):
        pos = ax.get_position()
        ax.set_position([pos.x0 + .05, pos.y0 + .025 * i , pos.width - 0.05, pos.height])
    [sns.despine(ax = ax, left=False, bottom=False, right=True) for ax in ax_arr]

    ax_arr[-1].set_xlabel('Epoch number', fontsize = 12)
    ax_arr[0].legend()

def set_properties_assignment(video, fig, ax, number_of_accumulation_steps, zoom):
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.05, pos.y0, pos.width - 0.05, pos.height])
    sns.despine(ax = ax, left=False, bottom=False, right=True)
    ax.tick_params(axis='both', which='major', pad=5)

    axes_position_assignment = ax.get_position()
    text_axes = fig.add_axes([axes_position_assignment.x0 + axes_position_assignment.width + 0.025, axes_position_assignment.y0, 0.01, axes_position_assignment.height])
    text_axes.text(0.5, 0.5,'Residual \nidentification', horizontalalignment='center', verticalalignment='center', rotation=90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)

    ax.set_yticks(range(19,video.number_of_animals+1,20))
    ax.set_yticklabels(range(20,video.number_of_animals+1,20))
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    ax.set_xlim((zoom[0]-50, zoom[1] + 50))
    ax.set_ylim([-.5, video.number_of_animals + .5 - 1])
    ax.set_ylabel('Individual', fontsize = 12)
    ax.set_xlabel('Frame number', fontsize = 12)

    ax.add_patch(patches.Rectangle(
            (zoom[0], 0 - 0.5),   # (x,y)
            zoom[1] - zoom[0] + 1,  # width
            video.number_of_animals,          # height
            fill = False,
            edgecolor = 'k',
            linewidth = 2.,
            linestyle = '--',
            facecolor = None))

def set_properties_final_accuracy(video, fig, ax_P2, ax_gt_accuracy, number_of_accumulation_steps):
    pos = ax_P2.get_position()
    ax_P2.set_position([pos.x0 + .05, pos.y0, pos.width - 0.05, pos.height - 0.025])
    ax_P2.set_xticklabels([])
    axes_position_assignment = ax_P2.get_position()
    text_axes = fig.add_axes([axes_position_assignment.x0 + axes_position_assignment.width + 0.025, axes_position_assignment.y0, 0.01, axes_position_assignment.height])
    text_axes.text(0.5, 0.5,'Estimated', horizontalalignment='center', verticalalignment='center', rotation=90, fontsize = 12)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)

    pos = ax_gt_accuracy.get_position()
    ax_gt_accuracy.set_position([pos.x0 + .05, pos.y0 + 0.04, pos.width - 0.05, pos.height - 0.025])
    axes_position_assignment = ax_gt_accuracy.get_position()
    text_axes = fig.add_axes([axes_position_assignment.x0 + axes_position_assignment.width + 0.025, axes_position_assignment.y0, 0.01, axes_position_assignment.height])
    text_axes.text(0.5, 0.5,'Groundtruth', horizontalalignment='center', verticalalignment='center', rotation=90, fontsize = 12)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)

    ax_gt_accuracy.set_xlabel('Accuracy', fontsize = 12)
    ax_gt_accuracy.tick_params(axis='both', which='major', pad=5)

    fig.text(pos.x0, pos.y0 + 0.05, 'Number of individuals', ha='left', va='bottom', rotation='vertical', fontsize = 12)
    [ax.set_xlim((ax.get_xlim()[0], 100)) for ax in [ax_P2, ax_gt_accuracy]]
    [sns.despine(ax = ax, left=False, bottom=False, right=True) for ax in [ax_P2, ax_gt_accuracy]]

def set_properties_crossigns(video, fig, ax, number_of_accumulation_steps, zoom):
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.05, pos.y0 + .015, pos.width - 0.05, pos.height])
    sns.despine(ax = ax, left=False, bottom=False, right=True)
    ax.tick_params(axis='both', which='major', pad=5)

    axes_position_assignment = ax.get_position()
    text_axes = fig.add_axes([axes_position_assignment.x0 + axes_position_assignment.width + 0.025, axes_position_assignment.y0, 0.01, axes_position_assignment.height])
    text_axes.text(0.5, 0.5,'Crossing \nsolving', horizontalalignment='center', verticalalignment='center', rotation=90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)

    ax.set_yticks(range(19,video.number_of_animals+1,20))
    ax.set_yticklabels(range(20,video.number_of_animals+1,20))
    # ax.set_yticklabels([])
    ax.set_xlim((zoom[0]-50, zoom[1] + 50))
    ax.set_ylim([-.5, video.number_of_animals + .5 - 1])
    ax.set_xlabel('Frame number', fontsize = 12)
    ax.set_ylabel('Individual', fontsize = 12)

    ax.add_patch(patches.Rectangle(
            (zoom[0], 0 - 0.5),   # (x,y)
            zoom[1] - zoom[0] + 1,  # width
            video.number_of_animals,          # height
            fill = False,
            edgecolor = 'k',
            linewidth = 2.,
            linestyle = '--',
            facecolor = None))

def get_list_of_accumulation_steps_to_plot(number_of_accumulation_steps):
    if number_of_accumulation_steps <= 4:
        return range(number_of_accumulation_steps)
    elif number_of_accumulation_steps > 4:
        list_of_accumulation_steps = [0, 1]
        intermidiate_accumulation_step = int(np.ceil((number_of_accumulation_steps-2)/2) + 2)
        list_of_accumulation_steps.append(intermidiate_accumulation_step)
        list_of_accumulation_steps.append(number_of_accumulation_steps-1)
        return list_of_accumulation_steps

def get_identity_to_blob_hierarchy_list(list_of_fragments):
    return [fragment.blob_hierarchy_in_starting_frame
            for fragment in list_of_fragments.fragments
            if fragment.accumulation_step == 0]

def get_no_gaps_fragments(list_of_blobs_no_gaps, number_of_animals):
    no_gaps_fragments = {i:[] for i in range(1, number_of_animals + 1)}
    temp_fragments = {i:[None, None] for i in range(1, number_of_animals + 1)}
    for frame_number, blobs_in_frame in enumerate(list_of_blobs_no_gaps.blobs_in_video):
        identities_in_frame = list(flatten([blob.assigned_identity for blob in blobs_in_frame]))
        for identity in range(1, number_of_animals + 1):
            if identity in identities_in_frame and temp_fragments[identity][0] is None:
                temp_fragments[identity][0] = frame_number
            elif identity in identities_in_frame and temp_fragments[identity][0] is not None and frame_number == len(list_of_blobs_no_gaps.blobs_in_video) - 1:
                temp_fragments[identity][1] = frame_number
                no_gaps_fragments[identity].append(temp_fragments[identity])
            elif identity not in identities_in_frame and temp_fragments[identity][0] is not None:
                temp_fragments[identity][1] = frame_number
                no_gaps_fragments[identity].append(temp_fragments[identity])
                temp_fragments[identity] = [None, None]

    return no_gaps_fragments

def plot_accumulation_steps(video, list_of_fragments, list_of_blobs, list_of_blobs_no_gaps, training_dict, validation_dict):
    no_gaps_fragments = get_no_gaps_fragments(list_of_blobs_no_gaps, video.number_of_animals)
    interpolated_fragments = get_no_gaps_fragments(list_of_blobs, video.number_of_animals)

    plt.ion()
    sns.set_style("ticks")
    zoomed_frames = (1348, 1700)
    number_of_accumulation_steps = get_number_of_accumulation_steps(list_of_fragments)
    # identity_to_blob_hierarchy_list = get_identity_to_blob_hierarchy_list(list_of_fragments)
    identity_to_blob_hierarchy_list = range(video.number_of_animals+1)

    # list_of_accumulation_steps = get_list_of_accumulation_steps_to_plot(number_of_accumulation_steps)
    my_dpi = 96
    list_of_accumulation_steps = [0, 1, 2, number_of_accumulation_steps - 1]
    fig1 = plt.figure(figsize=(13, 10), dpi=my_dpi)
    colors = get_spaced_colors_util(video.number_of_animals, norm=True, black=False)

    ### Deep fingerprinting
    ax_arr_fragments = []
    print("list_of_accumulation_steps ", list_of_accumulation_steps)
    number_of_accumulation_steps_to_plot = len(list_of_accumulation_steps)
    for i, accumulation_step in enumerate(list_of_accumulation_steps):
        ax = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (i, 0), colspan=3)
        ax_arr_fragments.append(ax)
        ax.add_patch(patches.Rectangle((0, 0 - 0.5),   # (x,y)
                                        video.number_of_frames-1,  # width
                                        video.number_of_animals, # height
                                        fill = True,
                                        facecolor = 'k',
                                        edgecolor = 'None'))
        plot_accumulation_step_from_fragments(list_of_fragments.fragments, ax, accumulation_step, False, colors, identity_to_blob_hierarchy_list)

    ### Zoomed dentification
    ax_zoomed_assignment = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (0, 3), colspan=2)
    ax_zoomed_assignment.add_patch(patches.Rectangle((0, 0 - 0.5),   # (x,y)
                                    video.number_of_frames-1,  # width
                                    video.number_of_animals, # height
                                    fill = True,
                                    facecolor = 'k',
                                    edgecolor = 'None'))
    plot_accumulation_step_from_fragments(list_of_fragments.fragments, ax_zoomed_assignment, accumulation_step, True, colors, identity_to_blob_hierarchy_list)
    # plot_crossings_step(interpolated_fragments, ax_zoomed_assignment, colors)

    ### crossing steps
    ax_crossings = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (3, 3), colspan=2)
    plot_crossings_step(no_gaps_fragments, ax_crossings, colors)

    ### Final certainty
    ax_P2 = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (1, 3), colspan=2)
    ax_gt_accuracy = plt.subplot2grid((number_of_accumulation_steps_to_plot + 2, 5), (2, 3), colspan=2)
    plot_individual_certainty_and_accuracy(video, ax_P2, ax_gt_accuracy, colors, identity_to_blob_hierarchy_list)

    ### settings for figures and axes
    fig1.subplots_adjust(left=None, bottom=.15, right=None, top=.95,
                wspace=.2, hspace=.5)
    set_properties_fragments(video, fig1, ax_arr_fragments, number_of_accumulation_steps_to_plot, list_of_accumulation_steps, zoomed_frames = zoomed_frames)

    set_properties_assignment(video, fig1, ax_zoomed_assignment, number_of_accumulation_steps_to_plot, zoom = zoomed_frames)
    set_properties_final_accuracy(video, fig1, ax_P2, ax_gt_accuracy, number_of_accumulation_steps_to_plot)

    set_properties_crossigns(video, fig1, ax_crossings, number_of_accumulation_steps, zoom = zoomed_frames)

    # fig1.savefig(os.path.join(video.session_folder,'accumulation_steps_1.pdf'), transparent=True, dpi = 600)
    fig1.savefig(os.path.join(video.session_folder,'accumulation_steps_1.png'), transparent=False, dpi = 600)
    plt.show()

def update_fragments_with_groundtruth_identities(video, list_of_fragments, groundtruth):

    for blobs_in_frame in groundtruth.blobs_in_video:
        for blob_gt in blobs_in_frame:
            if blob_gt.is_an_individual:
                fragment = list_of_fragments.fragments[video.fragment_identifier_to_index[blob_gt.fragment_identifier]]
                fragment._user_generated_identity = blob_gt.identity
    return list_of_fragments

if __name__ == '__main__':
    path_to_results_hard_drive = '/media/atlas/ground_truth_results_backup'
    session_path = 'tracked_videos/idTrackerDeep_LargeGroups_3/100fish/First/session_02122017'
    path_to_video_object = os.path.join(path_to_results_hard_drive, session_path, 'video_object.npy')
    print("loading video object")
    video = np.load(path_to_video_object).item(0)
    video.update_paths(path_to_video_object)
    print("loading groundtruth")
    groundtruth = np.load(os.path.join(video.video_folder, '_groundtruth.npy')).item()
    print("loading list_of_blobs_no_gaps")
    list_of_blobs_no_gaps = ListOfBlobs.load(video, video.blobs_no_gaps_path)
    print("loading list_of_blobs")
    list_of_blobs = ListOfBlobs.load(video, video.blobs_path)
    print("loading light_list_of_fragments")
    list_of_fragments_dictionaries = np.load(os.path.join(video._accumulation_folder,'light_list_of_fragments.npy'))
    fragments = [Fragment(number_of_animals = video.number_of_animals) for fragment_dictionary in list_of_fragments_dictionaries]
    [fragment.__dict__.update(fragment_dictionary) for fragment, fragment_dictionary in zip(fragments, list_of_fragments_dictionaries)]
    light_list_of_fragments = ListOfFragments(fragments)
    light_list_of_fragments = update_fragments_with_groundtruth_identities(video, light_list_of_fragments, groundtruth)
    print("loading training and validation dictionaries")
    training_dict = np.load(os.path.join(video.accumulation_folder, 'training_loss_acc_dict.npy')).item()
    validation_dict = np.load(os.path.join(video.accumulation_folder, 'validation_loss_acc_dict.npy')).item()
    print("plotting")
    plot_accumulation_steps(video, light_list_of_fragments, list_of_blobs, list_of_blobs_no_gaps, training_dict, validation_dict)
