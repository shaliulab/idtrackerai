from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./')
sys.path.append('./utils')
sys.path.append('./library')
sys.path.append('./network/identification_model')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from pprint import pprint

from video import Video
from list_of_global_fragments import ListOfGlobalFragments

def add_subplot_axes(fig, ax, rect, axisbg='w'):
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def get_number_of_images_in_shortest_fragment_in_first_global_fragment(list_of_global_fragments, video):
    if hasattr(video, 'accumulation_folder'):
        list_of_global_fragments.order_by_distance_travelled()
        global_fragment_for_accumulation = int(video.accumulation_folder[-1])
        if global_fragment_for_accumulation > 0:
            global_fragment_for_accumulation -= 1

        number_of_images_in_fragments = list_of_global_fragments.global_fragments[global_fragment_for_accumulation].number_of_images_per_individual_fragment
        return np.min(number_of_images_in_fragments)
    else:
        return None

def plot_minimum_number_of_images_figure(fig_num_images_accuracy, ax_arr_num_images_accuracy, \
                                            tracked_videos_data_frame, group_sizes_list, \
                                            accuracies, images_in_shortest_fragment_in_first_global_fragment, protocols_array):
    accuracies = accuracies*100
    for i, group_size in enumerate(group_sizes_list):
        j = 0 if group_size <= 10 else 1
        all_accuracies = np.ravel(accuracies[i,:,:,:])
        minimum_number_of_images = np.ravel(images_in_shortest_fragment_in_first_global_fragment[i,:,:,:])
        protocols = np.ravel(protocols_array[i,:,:,:])
        for number_of_images, accuracy, protocol in zip(minimum_number_of_images, all_accuracies, protocols):
            if protocol == 1:
                marker = '^'
            elif protocol == 2:
                marker = 'o'
            elif protocol == 3:
                marker = 's'
            ax_arr_num_images_accuracy[j].semilogx(number_of_images, accuracy, alpha = 1,
                                                markeredgecolor = 'k', markeredgewidth=1,
                                                marker = marker, markerfacecolor = 'None')


    for i in range(len(tracked_videos_data_frame)):
        species = tracked_videos_data_frame.loc[i].animal_type
        if 'zebrafish' in species or 'drosophila' in species:
            if 'zebrafish' in species:
                color = 'g'
            elif 'drosophila' in species:
                color = 'm'
            accuracy = tracked_videos_data_frame.loc[i].accuracy_identification_and_interpolation * 100
            if tracked_videos_data_frame.loc[i].minimum_number_of_frames_moving_in_first_global_fragment is not None:
                minimum_number_of_images = tracked_videos_data_frame.loc[i].minimum_number_of_frames_moving_in_first_global_fragment
            else:
                minimum_number_of_images = tracked_videos_data_frame.loc[i].number_of_images_in_shortest_fragment_in_first_global_fragment
            group_size = tracked_videos_data_frame.loc[i].number_of_animals
            protocol = tracked_videos_data_frame.loc[i].protocol_used
            if protocol == 1:
                marker = '^'
            elif protocol == 2:
                marker = 'o'
            elif protocol == 3:
                marker = 's'
            j = 0 if group_size <= 10 else 1
            ax_arr_num_images_accuracy[j].semilogx(minimum_number_of_images, accuracy, alpha = 1.,
                                                        marker = marker, markerfacecolor = color,
                                                        markersize = 10)

    ax_arr_num_images_accuracy[0].axvline(30, c = 'r', ls = '--')
    ax_arr_num_images_accuracy[1].axvline(30, c = 'r', ls = '--')


def set_minimum_number_of_images_figure(fig_num_images_accuracy, ax_arr_num_images_accuracy):
    ax_arr_num_images_accuracy[0].set_title('Group size ' + r'$\leq$ 10', fontsize = 22, y = 1.05)
    ax_arr_num_images_accuracy[1].set_title('Group size ' + r'$\geq$ 38', fontsize = 22, y = 1.05)
    ax_arr_num_images_accuracy[0].set_xlabel('Number of training images', fontsize = 20)
    ax_arr_num_images_accuracy[0].set_ylabel('Accuracy', fontsize = 20)
    ax_arr_num_images_accuracy[1].set_xlabel('Number of images', fontsize = 20)
    ax_arr_num_images_accuracy[1].tick_params(axis='both', which='major', labelsize=14)
    ax_arr_num_images_accuracy[0].tick_params(axis='both', which='major', labelsize=14)
    ax_arr_num_images_accuracy[1].set_yticklabels([])
    ax_arr_num_images_accuracy[0].set_xticks([10, 100, 1000])
    ax_arr_num_images_accuracy[1].set_xticks([10, 100, 1000])
    ax_arr_num_images_accuracy[0].set_xticklabels([10, 100, 1000])
    ax_arr_num_images_accuracy[1].set_xticklabels([10, 100, 1000])
    ax_arr_num_images_accuracy[0].set_ylim((0,102))
    ax_arr_num_images_accuracy[1].set_ylim((0,102))


    sns.despine(ax = ax_arr_num_images_accuracy[0], right = True, top = True)
    sns.despine(ax = ax_arr_num_images_accuracy[1], right = True, top = True)

    simulated_videos = mpatches.Patch(color='k', fc = 'None', linewidth = 1, label='Simulated videos')
    real_videos = mpatches.Patch(color='k', alpha = 1., label='Real videos')
    fish_videos = mpatches.Patch(color='g', alpha = 1., label='Zebrafish')
    flies_videos = mpatches.Patch(color='m', alpha = 1., label='Drosophila')
    protocol_1 = mlines.Line2D([], [], color='k', marker='^', markersize=6, label='Protocol 1',
                                markeredgecolor = 'k', markeredgewidth=1, markerfacecolor='None',
                                linestyle = 'None')
    protocol_2 = mlines.Line2D([], [], color='k', marker='o', markersize=6, label='Protocol 2',
                                markeredgecolor = 'k', markeredgewidth=1, markerfacecolor='None',
                                linestyle = 'None')
    protocol_3 = mlines.Line2D([], [], color='k', marker='s', markersize=6, label='Protocol 3',
                                markeredgecolor = 'k', markeredgewidth=1, markerfacecolor='None',
                                linestyle = 'None')


    ax_arr_num_images_accuracy[0].legend(handles=[simulated_videos, real_videos,
                                                fish_videos, flies_videos,
                                                protocol_1,
                                                protocol_2,
                                                protocol_3], loc = 4)

if __name__ == '__main__':
    tracked_videos_data_frame = pd.read_pickle('/media/chronos/ground_truth_results_backup/tracked_videos_data_frame.pkl')
    ### load global results data frame
    if os.path.isfile('./library/library_test_algorithm_test_GHI_aaa_cnn_0/results_data_frame.pkl'):
        print("loading results_data_frame.pkl...")
        results_data_frame = pd.read_pickle('./library/library_test_algorithm_test_GHI_aaa_cnn_0/results_data_frame.pkl')
        print("results_data_frame.pkl loaded \n")
    else:
        print("results_data_frame.pkl does not exist \n")
    repetition_to_plot = int(sys.argv[1]) if int(sys.argv[1]) != 0 else None
    if repetition_to_plot is not None:
        results_data_frame = results_data_frame[results_data_frame.repetition == repetition_to_plot]

    # get tests_data_frame and test to plot
    print("loading tests data frame")
    tests_data_frame = pd.read_pickle('./library/library_test_algorithm_test_GHI_aaa_cnn_0/tests_data_frame.pkl')
    test_dictionary = tests_data_frame.loc[12].to_dict()
    frames_in_video = test_dictionary['frames_in_video'][0]

    pprint(test_dictionary)

    ### Initialize arrays
    group_sizes_list = test_dictionary['group_sizes']
    scale_parameter_list = test_dictionary['scale_parameter'][::-1]
    shape_parameter_list = test_dictionary['shape_parameter'][::-1]
    number_of_group_sizes = len(group_sizes_list)
    number_of_scale_values = len(scale_parameter_list)
    number_of_shape_values = len(shape_parameter_list)
    number_of_repetitions = len(results_data_frame.repetition.unique())
    protocol = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    accuracy = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    images_in_shortest_fragment_in_first_global_fragment = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))

    plt.ion()
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    plt.close()
    sns.set_style("ticks")
    for i, group_size in enumerate(group_sizes_list):
        print("***group_size ", group_size)
        for j, scale_parameter in enumerate(scale_parameter_list):
            if scale_parameter % 1 == 0:
                scale_parameter = int(scale_parameter)

            for k, shape_parameter in enumerate(shape_parameter_list):
                print('----- ', scale_parameter, shape_parameter)
                if shape_parameter % 1 == 0:
                    shape_parameter = int(shape_parameter)

                for l, repetition in enumerate(results_data_frame.repetition.unique()):
                    repetition_path = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0],
                                                    'group_size_' + str(int(group_size)),
                                                    'num_frames_' + str(int(frames_in_video)),
                                                    'scale_parameter_' + str(scale_parameter),
                                                    'shape_parameter_' + str(shape_parameter),
                                                    'repetition_' + str(int(repetition)))
                    try:
                        video_path = os.path.join(repetition_path, 'session', 'video_object.npy')
                        video = np.load(video_path).item(0)
                        video_object_found = True
                    except:
                        video_object_found = False
                        print("video object not found")

                    try:
                        list_of_global_fragments = np.load(video.global_fragments_path).item()
                        list_of_global_fragments_found = True
                    except:
                        list_of_global_fragments_found = False

                    ### Get data for repetition
                    results_data_frame_rep = results_data_frame.query('group_size == @group_size' +
                                                                ' & frames_in_video == @frames_in_video' +
                                                                ' & scale_parameter == @scale_parameter' +
                                                                ' & shape_parameter == @shape_parameter' +
                                                                ' & repetition == @repetition')
                    if results_data_frame_rep.protocol.item() is None:
                        video_object_found = False
                        print("Algorithm did not finish")

                    ### Get statistics
                    if len(results_data_frame_rep) != 0:
                        protocol[i,k,j,l] = results_data_frame_rep.protocol.item() if video_object_found else None
                        accuracy[i,k,j,l] = results_data_frame_rep.accuracy.item() if video_object_found else None
                        images_in_shortest_fragment_in_first_global_fragment[i,k,j,l] = get_number_of_images_in_shortest_fragment_in_first_global_fragment(list_of_global_fragments, video)

    ### plot minimun number of images in first global fragment vs accuracy
    fig_num_images_accuracy, ax_arr_num_images_accuracy = plt.subplots(1,2, sharey = False, sharex = False)
    fig_num_images_accuracy.set_size_inches((screen_x/100*2/3,screen_y/100*5/8))
    plot_minimum_number_of_images_figure(fig_num_images_accuracy,
                                        ax_arr_num_images_accuracy,
                                        tracked_videos_data_frame,
                                        group_sizes_list, accuracy,
                                        images_in_shortest_fragment_in_first_global_fragment,
                                        protocol)
    set_minimum_number_of_images_figure(fig_num_images_accuracy, ax_arr_num_images_accuracy)

    path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0])
    fig_num_images_accuracy.savefig(os.path.join(path_to_save_figure, 'number_of_images_vs_accuracy.pdf'), transparent = True)

    plt.show()
