from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('../')
sys.path.append('./utils')
sys.path.append('./library')

import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, is_color_like
import seaborn as sns
import pandas as pd
from pprint import pprint

from py_utils import get_spaced_colors_util
from library_utils import LibraryJobConfig

if __name__ == '__main__':

    ### load global results data frame
    if os.path.isfile('./library/results_data_frame.pkl'):
        print("loading results_data_frame.pkl...")
        results_data_frame = pd.read_pickle('./library/results_data_frame.pkl')
        print("results_data_frame.pkl loaded \n")
    else:
        print("results_data_frame.pkl does not exist \n")
    results_data_frame = results_data_frame[results_data_frame.repetition == 1]

    # get tests_data_frame and test to plot
    print("loading tests data frame")
    tests_data_frame = pd.read_pickle('./library/tests_data_frame.pkl')
    test_dictionary = tests_data_frame.loc[12].to_dict()

    pprint(test_dictionary)

    ### Initialize arrays
    scale_list = test_dictionary['scale']
    shape_list = test_dictionary['shape']
    number_of_conditions_mean = len(results_data_frame.loc[:,'scale'].unique())
    number_of_condition_var = len(results_data_frame.loc[:,'shape'].unique())
    number_of_repetitions = len(results_data_frame.repetition.unique())
    protocol = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    total_time = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    ratio_of_accumulated_images = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    ratio_of_video_accumulated = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    overall_P2 = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    accuracy = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    accuracy_in_accumulation = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    accuracy_after_accumulation = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))

    plt.ion()
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    sns.set_style("ticks")
    fig_distributions_list = []
    fig_statistics_list = []
    for group_size in results_data_frame.group_size.unique():

        for frames_in_video in results_data_frame.frames_in_video.unique():
            fig_distributions, ax_arr = plt.subplots(len(results_data_frame.loc[:,'scale'].unique()), len(results_data_frame.loc[:,'shape'].unique()),
                                        sharex = True, sharey = False)
            fig_distributions_list.append(fig_distributions)
            fig_distributions.suptitle('Group size %i - Frames in video %i' %(group_size, frames_in_video))


            for i, scale in enumerate(scale_list):
                if scale % 1 == 0:
                    scale = int(scale)

                for j, shape in enumerate(shape_list):
                    if shape % 1 == 0:
                        shape = int(shape)

                    for k, repetition in enumerate(results_data_frame.repetition.unique()):
                        repetition_path = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0],
                                                        'group_size_' + str(int(group_size)),
                                                        'num_frames_' + str(int(frames_in_video)),
                                                        'scale_' + str(scale),
                                                        'shape_' + str(shape),
                                                        'repetition_' + str(int(repetition)))
                        video_path = os.path.join(repetition_path, 'session', 'video_object.npy')
                        video = np.load(video_path).item(0)
                        ### Plot distributions
                        if repetition == 1:
                            nbins = 10
                            ax = ax_arr[j,i]
                            number_of_images_in_individual_fragments = video.individual_fragments_distance_travelled[0]
                            number_of_images_in_individual_fragments = number_of_images_in_individual_fragments[number_of_images_in_individual_fragments > 3]
                            MIN = np.min(number_of_images_in_individual_fragments)
                            MAX = np.max(number_of_images_in_individual_fragments)
                            hist, bin_edges = np.histogram(number_of_images_in_individual_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
                            ax.semilogx(bin_edges[:-1], hist, '-ob' ,markersize = 5)
                            if j == 3:
                                ax.set_xlabel('number of frames \nscale = %.1f' %scale)
                            if i == 0:
                                ax.set_ylabel('shape = %.1f \nnumber of \nindividual \nfragments' %shape)
                            mean = shape * scale
                            sigma = np.sqrt(shape * scale**2)
                            title = r'$\mu$ = %.2f, $\sigma$ = %.2f' %(mean, sigma)
                            ax.set_title(title)

                        ### Create accuracy matrix
                        results_data_frame_rep = results_data_frame.query('group_size == @group_size' +
                                                                    ' & frames_in_video == @frames_in_video' +
                                                                    ' & scale == @scale' +
                                                                    ' & shape == @shape' +
                                                                    ' & repetition == @repetition')
                        protocol[j,i,k] = results_data_frame_rep.protocol
                        total_time[j,i,k] = results_data_frame_rep.total_time
                        ratio_of_accumulated_images[j,i,k] = (results_data_frame_rep.number_of_partially_accumulated_individual_blobs
                                                                + results_data_frame_rep.number_of_globally_accumulated_individual_blobs) / \
                                                                (results_data_frame_rep.number_of_blobs -
                                                                results_data_frame_rep.number_of_not_accumulable_individual_blobs)
                        ratio_of_video_accumulated[j,i,k] = (results_data_frame_rep.number_of_partially_accumulated_individual_blobs
                                                                + results_data_frame_rep.number_of_globally_accumulated_individual_blobs) / \
                                                                results_data_frame_rep.number_of_blobs
                        overall_P2[j,i,k] = video.overall_P2
                        accuracy[j,i,k] = results_data_frame_rep.accuracy
                        accuracy_in_accumulation[j,i,k] = results_data_frame_rep.accuracy_in_accumulation
                        accuracy_after_accumulation[j,i,k] = results_data_frame_rep.accuracy_after_accumulation

            fig_statistics, ax_arr = plt.subplots(2,4)
            fig_statistics_list.append(fig_statistics)
            fig_statistics.suptitle('Group size %i - Frames in video %i' %(group_size, frames_in_video))

            def plot_statistics_heatmap(ax, matrix, title, xticklabels, yticklabels, vmax = None, vmin = None):
                ax = sns.heatmap(np.mean(matrix, axis = 2),
                                    ax = ax,
                                    fmt = '.3f',
                                    square = True,
                                    cbar = False,
                                    xticklabels = xticklabels,
                                    yticklabels = yticklabels,
                                    vmax = vmax,
                                    vmin = vmin,
                                    annot=True)
                ax.set_title(title)

            plot_statistics_heatmap(ax_arr[0,0], protocol, 'protocol', scale_list, shape_list)
            plot_statistics_heatmap(ax_arr[0,1], total_time, 'total time', scale_list, shape_list)
            plot_statistics_heatmap(ax_arr[0,2], ratio_of_accumulated_images, r'$\%$' + ' accumulated images', scale_list, shape_list, vmax = 1, vmin = 0 )
            plot_statistics_heatmap(ax_arr[0,3], ratio_of_video_accumulated, r'$\%$' + ' video', scale_list, shape_list, vmax = 1, vmin = 0)
            plot_statistics_heatmap(ax_arr[1,0], overall_P2, 'overall P2', scale_list, shape_list, vmax = 1, vmin = 0)
            plot_statistics_heatmap(ax_arr[1,1], accuracy, 'accuracy', scale_list, shape_list, vmax = 1, vmin = 0)
            plot_statistics_heatmap(ax_arr[1,2], accuracy_in_accumulation, 'accuracy in accumulation', scale_list, shape_list, vmax = 1, vmin = 0)
            plot_statistics_heatmap(ax_arr[1,3], accuracy_after_accumulation, 'accuracy after accumulation', scale_list, shape_list, vmax = 1, vmin = 0)

            ax_arr[0,0].set_ylabel('shape')
            ax_arr[0,1].set_xticklabels([]), ax_arr[0,1].set_yticklabels([])
            ax_arr[0,2].set_xticklabels([]), ax_arr[0,2].set_yticklabels([])
            ax_arr[0,3].set_xticklabels([]), ax_arr[0,3].set_yticklabels([])
            ax_arr[1,0].set_xlabel('scale')
            ax_arr[1,0].set_ylabel('shape')
            ax_arr[1,1].set_xlabel('scale')
            ax_arr[1,1].set_yticklabels([])
            ax_arr[1,2].set_xlabel('scale')
            ax_arr[1,2].set_yticklabels([])
            ax_arr[1,3].set_xlabel('scale')
            ax_arr[1,3].set_yticklabels([])

    path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0],
                                    'group_size_' + str(int(group_size)))



    [fig.set_size_inches((screen_x/100,screen_y/100)) for fig in fig_statistics_list + fig_distributions_list]
    [fig.savefig(os.path.join(path_to_save_figure, 'distributions_%i.pdf' %video_length), transparent = True) for video_length, fig in zip(results_data_frame.frames_in_video.unique(), fig_distributions_list)]
    [fig.savefig(os.path.join(path_to_save_figure, 'statistics_%i.pdf' %video_length), transparent = True) for video_length, fig in zip(results_data_frame.frames_in_video.unique(), fig_statistics_list)]

    plt.show()
