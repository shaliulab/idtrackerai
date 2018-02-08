from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./')
sys.path.append('./utils')
sys.path.append('./library')
sys.path.append('./network/identification_model')

import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, is_color_like
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from pprint import pprint
from scipy.stats import gamma
from scipy import stats

from py_utils import get_spaced_colors_util
from video import Video
from globalfragment import GlobalFragment
from list_of_global_fragments import ListOfGlobalFragments

def pdf2logpdf(pdf):
    def logpdf(x):
        return pdf(x)*x*np.log(10)
    return logpdf

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

def plot_statistics_heatmap(ax, matrix, title, xticklabels, yticklabels, vmax = None, vmin = None, annot = True):
    if title == 'total time':
        fmt = '.1f'
    elif title == 'protocol-main' or title == 'accuracy-main':
        fmt = 's'
    elif title == 'protocol':
        fmt = '.0f'
    else:
        fmt = '.2f'
    ax = sns.heatmap(matrix,
                        ax = ax,
                        fmt = fmt,
                        square = True,
                        cbar = False,
                        xticklabels = xticklabels,
                        yticklabels = yticklabels,
                        vmax = vmax,
                        vmin = vmin,
                        annot = annot)
    ax.set_title(title)

def plot_all_statistics_figure(ax_arr, protocol, total_time, ratio_of_accumulated_images,
                            ratio_of_video_accumulated, overall_P2,
                            accuracy, accuracy_in_accumulation,
                            accuracy_after_accumulation,
                            scale_parameter_list, shape_parameter_list):
    protocol = np.nanmean(protocol, axis = 2)
    total_time = np.nanmean(total_time, axis = 2)
    ratio_of_accumulated_images = np.nanmean(ratio_of_accumulated_images, axis = 2)*100
    ratio_of_video_accumulated = np.nanmean(ratio_of_video_accumulated, axis = 2)*100
    overall_P2 = np.nanmean(overall_P2, axis = 2)*100
    accuracy = np.nanmean(accuracy, axis = 2)*100
    accuracy_in_accumulation = np.nanmean(accuracy_in_accumulation, axis = 2)*100
    accuracy_after_accumulation = np.nanmean(accuracy_after_accumulation, axis = 2)*100

    plot_statistics_heatmap(ax_arr[0,0], protocol, 'protocol', scale_parameter_list, shape_parameter_list, vmin = 1, vmax = 3)
    plot_statistics_heatmap(ax_arr[0,1], total_time, 'total time', scale_parameter_list, shape_parameter_list)
    plot_statistics_heatmap(ax_arr[0,2], ratio_of_accumulated_images, r'$\%$' + ' accumulated images', scale_parameter_list, shape_parameter_list, vmax = 100, vmin = 0 )
    plot_statistics_heatmap(ax_arr[0,3], ratio_of_video_accumulated, r'$\%$' + ' video', scale_parameter_list, shape_parameter_list, vmax = 100, vmin = 0)
    plot_statistics_heatmap(ax_arr[1,0], overall_P2, 'overall P2', scale_parameter_list, shape_parameter_list, vmax = 100, vmin = 0)
    plot_statistics_heatmap(ax_arr[1,1], accuracy, 'accuracy', scale_parameter_list, shape_parameter_list, vmax = 100, vmin = 0)
    plot_statistics_heatmap(ax_arr[1,2], accuracy_in_accumulation, 'accuracy in accumulation', scale_parameter_list, shape_parameter_list, vmax = 100, vmin = 0)
    plot_statistics_heatmap(ax_arr[1,3], accuracy_after_accumulation, 'accuracy after accumulation', scale_parameter_list, shape_parameter_list, vmax = 100, vmin = 0)

def set_properties_all_statistics_figure(ax_arr):
    ax_arr[0,0].set_ylabel('shape\n')
    ax_arr[0,0].set_xticklabels([])
    ax_arr[0,1].set_xticklabels([]), ax_arr[0,1].set_yticklabels([])
    ax_arr[0,2].set_xticklabels([]), ax_arr[0,2].set_yticklabels([])
    ax_arr[0,3].set_xticklabels([]), ax_arr[0,3].set_yticklabels([])
    ax_arr[1,0].set_xlabel('scale')
    ax_arr[1,0].set_ylabel('shape\n')
    ax_arr[1,1].set_xlabel('scale')
    ax_arr[1,1].set_yticklabels([])
    ax_arr[1,2].set_xlabel('scale')
    ax_arr[1,2].set_yticklabels([])
    ax_arr[1,3].set_xlabel('scale')
    ax_arr[1,3].set_yticklabels([])

def build_annotate_matrices(protocol, accuracy, accuracy_in_accumulation,
                            ratio_of_video_accumulated, ratio_of_accumulated_images):
    protocol_annotate = np.chararray(protocol.shape, itemsize = 15)
    accuracy_annotate = np.chararray(accuracy.shape, itemsize = 15)

    for group_size in range(protocol.shape[0]):
        for row in range(protocol.shape[1]):
            for column in range(protocol.shape[2]):
                if ratio_of_accumulated_images[group_size, row, column] >= 90:
                    protocol_annotate[group_size, row, column] = '%.0f' %protocol[group_size, row, column]
                    accuracy_annotate[group_size, row, column] = '%.2f' %accuracy[group_size, row, column]
                else:
                    protocol_annotate[group_size, row, column] = '%.0f ' %protocol[group_size, row, column]
                    accuracy_annotate[group_size, row, column] = '%.2f \n(%.2f)' %(accuracy_in_accumulation[group_size, row, column],
                                                                                    ratio_of_video_accumulated[group_size, row, column])
                    # accuracy_annotate[group_size, row, column] = '%.4f \n(%.3f)' %(accuracy[group_size, row, column], accuracy_in_accumulation[group_size, row, column])
                    accuracy[group_size, row, column] = accuracy_in_accumulation[group_size, row, column]


    return protocol, accuracy, protocol_annotate, accuracy_annotate

def plot_pair_of_statistics_per_group_sizes(fig_protocol_accuracy, ax_arr, stats1, stats2,
                                        group_sizes_list, scale_parameter_list, shape_parameter_list,
                                        title1, title2,
                                        min_max_1, min_max_2,
                                        annotate1 = True, annotate2 = True):

    for i, group_size in enumerate(group_sizes_list):
        plot_statistics_heatmap(ax_arr[0,i], stats1[i], title1,
                                scale_parameter_list, shape_parameter_list,
                                vmin = min_max_1[0], vmax = min_max_1[1],
                                annot = annotate1[i] if not isinstance(annotate1, bool) else True)
        plot_statistics_heatmap(ax_arr[1,i], stats2[i], title2,
                                scale_parameter_list, shape_parameter_list,
                                vmin = min_max_2[0], vmax = min_max_2[1],
                                annot = annotate2[i] if not isinstance(annotate2, bool) else True)

def set_protocol_accuracy_group_sizes(fig_protocol_accuracy, ax_arr, title1, title2):
    pos = ax_arr[0,0].get_position()
    text_axes = fig_protocol_accuracy.add_axes([pos.x0 - .05, pos.y0, 0.01, pos.height])
    text_axes.text(0.5, 0.5,title1, horizontalalignment='center', verticalalignment='center', rotation=90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)
    ax_arr[0,0].set_title('10 individuals')
    ax_arr[0,0].set_ylabel('shape')
    ax_arr[0,0].set_xticklabels([])
    ax_arr[0,1].set_title('60 individuals')
    ax_arr[0,1].set_xticklabels([])
    ax_arr[0,1].set_yticklabels([])
    ax_arr[0,2].set_title('100 individuals')
    ax_arr[0,2].set_xticklabels([])
    ax_arr[0,2].set_yticklabels([])

    pos = ax_arr[1,0].get_position()
    text_axes = fig_protocol_accuracy.add_axes([pos.x0 - .05, pos.y0, 0.01, pos.height])
    text_axes.text(0.5, 0.5,title2, horizontalalignment='center', verticalalignment='center', rotation=90, fontsize = 15)
    text_axes.set_xticks([])
    text_axes.set_yticks([])
    text_axes.grid(False)
    sns.despine(ax = text_axes, left=True, bottom=True, right=True)
    ax_arr[1,0].set_title('')
    ax_arr[1,0].set_ylabel('shape')
    ax_arr[1,0].set_xlabel('scale')
    ax_arr[1,1].set_title('')
    ax_arr[1,1].set_yticklabels([])
    ax_arr[1,1].set_xlabel('scale')
    ax_arr[1,2].set_title('')
    ax_arr[1,2].set_yticklabels([])
    ax_arr[1,2].set_xlabel('scale')


def plot_histogram_individual_fragments(ax, number_of_images_in_individual_fragments, scale_parameter, shape_parameter):
    gamma_simulation = gamma(shape_parameter, loc = 0.99, scale = scale_parameter)
    gamma_simulation_logpdf = pdf2logpdf(gamma_simulation.pdf)
    MIN = 1
    MAX = 10000
    logbins = np.linspace(np.log10(MIN), np.log10(MAX), nbins)
    ax.hist(np.log10(number_of_images_in_individual_fragments), bins = logbins, normed = True, label = 'simulated')
    logbins_pdf = np.linspace(np.log10(MIN), np.log10(MAX), 100)
    ax.plot(logbins_pdf, gamma_simulation_logpdf(np.power(10,logbins_pdf)), label = 'theoretical')
    if k == len(shape_parameter_list)-1:
        ax.set_xlabel('number of frames \n\nscale = %.2f' %scale_parameter)
    if j == 0:
        ax.set_ylabel('shape = %.2f \n\nPDF' %shape_parameter)
    th_mean = shape_parameter * scale_parameter
    th_sigma = np.sqrt(shape_parameter * scale_parameter**2)
    th_title = r'$\mu$ = %.2f, $\sigma$ = %.2f (th)' %(th_mean, th_sigma)
    mean = np.mean(number_of_images_in_individual_fragments)
    sigma = np.std(number_of_images_in_individual_fragments)
    title = r'$\mu$ = %.2f, $\sigma$ = %.2f (sim)' %(mean, sigma)
    ax.set_xlim((np.log10(MIN), np.log10(MAX)))
    ax.set_xticks([1,2,3])
    ax.set_xticklabels([10,100,1000])
    ax.text(2.25, 1., th_title, horizontalalignment = 'center')
    ax.text(2.25, 0.85, title, horizontalalignment = 'center')
    ax.set_ylim((0,1.3))
    if shape_parameter == 0.05 and scale_parameter == 100:
        ax.legend(loc = 7)
    sns.despine(ax = ax)

def get_number_of_images_in_shortest_fragment_in_first_global_fragment(list_of_global_fragments, video):
    if hasattr(video, 'accumulation_folder'):
        list_of_global_fragments.order_by_distance_travelled()
        global_fragment_for_accumulation = int(video.accumulation_folder[-1])
        if global_fragment_for_accumulation > 0:
            global_fragment_for_accumulation -= 1

        number_of_images_in_fragments = list_of_global_fragments.global_fragments[global_fragment_for_accumulation].number_of_images_per_individual_fragment
        print('minimum number of images ', np.min(number_of_images_in_fragments))
        return np.min(number_of_images_in_fragments)
    else:
        return None

def get_mean_number_of_images_in_first_global_fragment(list_of_global_fragments, video):
    if hasattr(video, 'accumulation_folder'):
        list_of_global_fragments.order_by_distance_travelled()
        global_fragment_for_accumulation = int(video.accumulation_folder[-1])
        if global_fragment_for_accumulation > 0:
            global_fragment_for_accumulation -= 1

        number_of_images_in_fragments = list_of_global_fragments.global_fragments[global_fragment_for_accumulation].number_of_images_per_individual_fragment
        return np.mean(number_of_images_in_fragments)
    else:
        return None

def plot_histogram_first_global_fragment_distribution(ax, list_of_global_fragments, video, scale_parameter, shape_parameter, accuracy, protocol):
    list_of_global_fragments.order_by_distance_travelled()
    global_fragment_for_accumulation = int(video.accumulation_folder[-1])
    if global_fragment_for_accumulation > 0:
        global_fragment_for_accumulation -= 1

    number_of_images_in_fragments = list_of_global_fragments.global_fragments[global_fragment_for_accumulation].number_of_images_per_individual_fragment
    MIN = 1
    MAX = 10000
    logbins = np.linspace(np.log10(MIN), np.log10(MAX), 25)
    ax.hist(np.log10(number_of_images_in_fragments), bins = logbins, normed = False)

    if k == len(shape_parameter_list)-1:
        ax.set_xlabel('number of frames \n\nscale = %.2f' %scale_parameter)
    if j == 0:
        ax.set_ylabel('shape = %.2f \n\nnumber of fragments' %shape_parameter)
    title_protocol_accuracy = 'Acc = %.2f, Protocol = %i' %(accuracy * 100, protocol)
    mean = np.mean(number_of_images_in_fragments)
    sigma = np.std(number_of_images_in_fragments)
    title_mean = r'$\mu$ = %.2f, $\sigma$ = %.2f' %(mean, sigma)
    ax.set_xlim((np.log10(MIN), np.log10(MAX)))
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels([1,10,100,1000])
    ax.text(1, np.ceil(ax.get_ylim()[1]*.95), title_protocol_accuracy, horizontalalignment = 'left')
    ax.text(1, np.floor(ax.get_ylim()[1]*.8), title_mean, horizontalalignment = 'left')
    sns.despine(ax = ax)

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
            # if j == 0:
            #     subax_smaller.semilogx(number_of_images, accuracy, alpha = .3, marker = marker, markerfacecolor = 'k')
            # elif j == 1:
            #     subax_larger.semilogx(number_of_images, accuracy, alpha = .3, marker = marker, markerfacecolor = 'k')


    for i in range(len(tracked_videos_data_frame)):
        species = tracked_videos_data_frame.loc[i].species
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
            protocol = tracked_videos_data_frame.loc[i].protocol
            if protocol == 1:
                marker = '^'
            elif protocol == 2:
                marker = 'o'
            elif protocol == 3:
                marker = 's'
            j = 0 if group_size <= 10 else 1
            ax_arr_num_images_accuracy[j].semilogx(minimum_number_of_images, accuracy, alpha = 1.,
                                                        marker = marker, markerfacecolor = color)
            # if j == 0:
            #     subax_smaller.semilogx(minimum_number_of_images, accuracy, alpha = 1, markeredgecolor=color, markeredgewidth=1, marker = marker, markerfacecolor = 'None')
            # elif j == 1:
            #     subax_larger.semilogx(minimum_number_of_images, accuracy, alpha = 1, markeredgecolor=color, markeredgewidth=1, marker = marker, markerfacecolor = 'None')
    ax_arr_num_images_accuracy[0].axvline(30, c = 'r', ls = '--')
    ax_arr_num_images_accuracy[1].axvline(30, c = 'r', ls = '--')
    # subax_smaller.axvline(30, c = 'r', ls = '--')
    # subax_larger.axvline(30, c = 'r', ls = '--')

def set_minimum_number_of_images_figure(fig_num_images_accuracy, ax_arr_num_images_accuracy):
    fig_num_images_accuracy.suptitle('Number of images in smaller fragment in the starting global fragment for accumulation', fontsize = 20)
    ax_arr_num_images_accuracy[0].set_title('Smaller groups', fontsize = 16)
    ax_arr_num_images_accuracy[1].set_title('Larger groups', fontsize = 16)
    ax_arr_num_images_accuracy[0].set_xlabel('Number of images', fontsize = 14)
    ax_arr_num_images_accuracy[0].set_ylabel('Accuracy', fontsize = 14)
    ax_arr_num_images_accuracy[1].set_xlabel('Number of images', fontsize = 14)
    ax_arr_num_images_accuracy[1].tick_params(axis='both', labelsize=14)
    ax_arr_num_images_accuracy[0].tick_params(axis='both', labelsize=14)
    ax_arr_num_images_accuracy[1].set_yticklabels([])
    ax_arr_num_images_accuracy[0].set_xticks([10, 100, 1000])
    ax_arr_num_images_accuracy[1].set_xticks([10, 100, 1000])
    ax_arr_num_images_accuracy[0].set_xticklabels([10, 100, 1000])
    ax_arr_num_images_accuracy[1].set_xticklabels([10, 100, 1000])
    ax_arr_num_images_accuracy[0].set_ylim((0,101))
    ax_arr_num_images_accuracy[1].set_ylim((0,101))
    ax_arr_num_images_accuracy[0].set_xlim((5,2200))
    ax_arr_num_images_accuracy[1].set_xlim((2,2200))

    # subax_smaller.tick_params(axis='both', labelsize=12)
    # subax_larger.tick_params(axis='both', labelsize=12)
    # subax_smaller.set_xticks([100, 1000])
    # subax_larger.set_xticks([10, 100])
    # subax_smaller.set_xticklabels([100, 1000])
    # subax_larger.set_xticklabels([10, 100])
    # subax_smaller.set_ylim((98.7,100.05))
    # subax_larger.set_xlim((7,500))
    # subax_larger.set_ylim((99.74,100.01))
    # sns.despine(ax = ax_arr_num_images_accuracy[0], right = True, top = True)
    # sns.despine(ax = ax_arr_num_images_accuracy[1], right = True, top = True)

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



def create_gamma_grid_figure(results_data_frame, share_x = True, share_y = True):
    fig_distributions, ax_arr = plt.subplots(len(results_data_frame.loc[:,'scale_parameter'].unique()), len(results_data_frame.loc[:,'shape_parameter'].unique()),
                                sharex = share_x, sharey = share_y)
    return fig_distributions, ax_arr

if __name__ == '__main__':
    tracked_videos_data_frame = pd.read_pickle('/media/chronos/ground_truth_results_backup/tracked_videos_data_frame.pkl')
    ### load global results data frame
    if os.path.isfile('./library/results_data_frame.pkl'):
        print("loading results_data_frame.pkl...")
        results_data_frame = pd.read_pickle('./library/results_data_frame.pkl')
        print("results_data_frame.pkl loaded \n")
    else:
        print("results_data_frame.pkl does not exist \n")
    repetition_to_plot = int(sys.argv[1]) if int(sys.argv[1]) != 0 else None
    if repetition_to_plot is not None:
        results_data_frame = results_data_frame[results_data_frame.repetition == repetition_to_plot]

    # get tests_data_frame and test to plot
    print("loading tests data frame")
    tests_data_frame = pd.read_pickle('./library/tests_data_frame.pkl')
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
    total_time = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    ratio_of_accumulated_images = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    ratio_of_video_accumulated = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    overall_P2 = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    accuracy = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    accuracy_in_accumulation = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    accuracy_after_accumulation = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    images_in_shortest_fragment_in_first_global_fragment = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))
    mean_number_of_iamgse_in_first_global_fragment = np.zeros((number_of_group_sizes, number_of_shape_values, number_of_scale_values, number_of_repetitions))

    plt.ion()
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    sns.set_style("ticks")
    for i, group_size in enumerate(group_sizes_list):
        print("***group_size ", group_size)
        fig_distributions, ax_arr = create_gamma_grid_figure(results_data_frame)
        fig_gf, ax_arr_gf = create_gamma_grid_figure(results_data_frame)
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

                    ### Create accuracy matrix
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
                        total_time[i,k,j,l] = results_data_frame_rep.total_time.item()  if video_object_found else None
                        ratio_of_accumulated_images[i,k,j,l] = (results_data_frame_rep.number_of_partially_accumulated_individual_blobs
                                                                + results_data_frame_rep.number_of_globally_accumulated_individual_blobs) / \
                                                                (video.individual_fragments_stats['number_of_accumulable_individual_blobs'])  if video_object_found else None
                        ratio_of_video_accumulated[i,k,j,l] = (results_data_frame_rep.number_of_partially_accumulated_individual_blobs
                                                                + results_data_frame_rep.number_of_globally_accumulated_individual_blobs) / \
                                                                results_data_frame_rep.number_of_blobs if video_object_found else None
                        overall_P2[i,k,j,l] = video.gt_accuracy['mean_individual_P2_in_validated_part'] if video_object_found else None
                        accuracy[i,k,j,l] = results_data_frame_rep.accuracy.item() if video_object_found else None
                        accuracy_in_accumulation[i,k,j,l] = results_data_frame_rep.accuracy_in_accumulation.item() if video_object_found else None
                        accuracy_after_accumulation[i,k,j,l] = results_data_frame_rep.accuracy_after_accumulation.item() if video_object_found else None
                        images_in_shortest_fragment_in_first_global_fragment[i,k,j,l] = get_number_of_images_in_shortest_fragment_in_first_global_fragment(list_of_global_fragments, video)
                        mean_number_of_iamgse_in_first_global_fragment[i,k,j,l] = get_mean_number_of_images_in_first_global_fragment(list_of_global_fragments, video)

                    # if l == 0:
                    #     ### Plot distributions
                    #     try:
                    #         nbins = 25
                    #         number_of_images_in_individual_fragments = results_data_frame_rep['individual_fragments_lengths'].item()
                    #         ax = ax_arr[k,j]
                    #         plot_histogram_individual_fragments(ax, number_of_images_in_individual_fragments, scale_parameter, shape_parameter)
                    #         fig_distributions.suptitle('Gamma distributions: Group size %i - repetition %i' %(group_size, repetition))
                    #     except:
                    #         number_of_images_in_individual_fragments = np.nan
                    #     ### Plot global fragments in 3 first global fragments
                    #     ax = ax_arr_gf[k,j]
                    #     fig_gf.suptitle("First global fragment distribution: Group size %i - repetition %i" %(group_size, repetition))
                    #     if list_of_global_fragments_found and video_object_found:
                    #         plot_histogram_first_global_fragment_distribution(ax, list_of_global_fragments, video,
                    #                                                             scale_parameter, shape_parameter,
                    #                                                             accuracy[i,k,j,l], protocol[i,k,j,l])
                    #     else:
                    #         if k == len(shape_parameter_list)-1:
                    #             ax.set_xlabel('number of frames \n\nscale = %.2f' %scale_parameter)
                    #         if j == 0:
                    #             ax.set_ylabel('shape = %.2f \n\nnumber of fragments' %shape_parameter)


        # # All statistics figure
        # fig_statistics, ax_arr = plt.subplots(2,4)
        # if repetition_to_plot is not None:
        #     fig_statistics.suptitle('Statistics: Group size %i - repetition %i' %(group_size, repetition_to_plot))
        # else:
        #     fig_statistics.suptitle('Statistics: Group size %i' %group_size)
        # plot_all_statistics_figure(ax_arr, protocol[i], total_time[i], ratio_of_accumulated_images[i],
        #                             ratio_of_video_accumulated[i], overall_P2[i],
        #                             accuracy[i], accuracy_in_accumulation[i],
        #                             accuracy_after_accumulation[i],
        #                             scale_parameter_list, shape_parameter_list)
        # set_properties_all_statistics_figure(ax_arr)
        #
        # # saving figures
        # path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0],
        #                             'group_size_' + str(int(group_size)))
        # if repetition_to_plot is not None:
        #     file_name_gamma_distributions = 'gamma_distributions_%i_repetition_%i.pdf' %(group_size, repetition_to_plot)
        #     file_name_statistics = 'statistics_%i_repetition_%i.pdf' %(group_size, repetition_to_plot)
        #     file_name_first_global_fragment_distribution = '1sGF_distribution_%i_repetition_%i.pdf' %(group_size, repetition_to_plot)
        # else:
        #     file_name_gamma_distributions = 'distributions_%i.pdf' %group_size
        #     file_name_statistics = 'statistics_%i.pdf' %group_size
        #     file_name_first_global_fragment_distribution = '1sGF_distribution_%i.pdf' %group_size
        # print(path_to_save_figure)
        # [fig.set_size_inches((screen_x/100,screen_y/100)) for fig in [fig_distributions, fig_statistics, fig_gf]]
        # fig_distributions.savefig(os.path.join(path_to_save_figure, file_name_gamma_distributions), transparent = True)
        # fig_statistics.savefig(os.path.join(path_to_save_figure, file_name_statistics), transparent = True)
        # fig_gf.savefig(os.path.join(path_to_save_figure, file_name_first_global_fragment_distribution), transparent = True)

    ### plot minimun number of images in first global fragment vs accuracy
    fig_num_images_accuracy, ax_arr_num_images_accuracy = plt.subplots(1,2, sharey = False, sharex = False)
    fig_num_images_accuracy.set_size_inches((screen_x/100*2/3,screen_y/100*3/4))
    # subax_smaller = add_subplot_axes(fig_num_images_accuracy, ax_arr_num_images_accuracy[0], [.55, .35, .4, .6], axisbg='w')
    # subax_larger = add_subplot_axes(fig_num_images_accuracy, ax_arr_num_images_accuracy[1], [.55, .35, .4, .6], axisbg='w')
    plot_minimum_number_of_images_figure(fig_num_images_accuracy,
                                        ax_arr_num_images_accuracy,
                                        tracked_videos_data_frame,
                                        group_sizes_list, accuracy,
                                        images_in_shortest_fragment_in_first_global_fragment,
                                        protocol)
    set_minimum_number_of_images_figure(fig_num_images_accuracy, ax_arr_num_images_accuracy)

    path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0])
    fig_num_images_accuracy.savefig(os.path.join(path_to_save_figure, 'number_of_imags_vs_accuracy.pdf'), transparent = True)

    # ### Compute average over repetitions
    # protocol = np.nanmean(protocol, axis = 3)
    # total_time = np.nanmean(total_time, axis = 3)*100
    # ratio_of_accumulated_images = np.nanmean(ratio_of_accumulated_images, axis = 3)*100
    # ratio_of_video_accumulated = np.nanmean(ratio_of_video_accumulated, axis = 3)*100
    # overall_P2 = np.nanmean(overall_P2, axis = 3)*100
    # accuracy = np.nanmean(accuracy, axis = 3)*100
    # accuracy_in_accumulation = np.nanmean(accuracy_in_accumulation, axis = 3)*100
    # accuracy_after_accumulation = np.nanmean(accuracy_after_accumulation, axis = 3)*100
    # ### main figure all group_sizes
    # # build annotate matrices
    # protocol, accuracy, \
    # protocol_annotate, accuracy_annotate = build_annotate_matrices(protocol, accuracy,
    #                                                             accuracy_in_accumulation,
    #                                                             ratio_of_video_accumulated,
    #                                                             ratio_of_accumulated_images)
    # fig, ax_arr = plt.subplots(2,3)
    # plt.subplots_adjust(left=.15, bottom=None, right=.85, top=None,
    #             wspace=.001, hspace=None)
    # plot_pair_of_statistics_per_group_sizes(fig, ax_arr,
    #                                     protocol, accuracy, group_sizes_list,
    #                                     scale_parameter_list, shape_parameter_list,
    #                                     'protocol-main', 'accuracy-main',
    #                                     [1,3], [0,100],
    #                                     protocol_annotate, accuracy_annotate)
    # fig.set_size_inches((screen_x/100,screen_y/100))
    # set_protocol_accuracy_group_sizes(fig, ax_arr,
    #                                 'protocol', 'accuracy')
    #
    # path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0])
    # if repetition_to_plot is not None:
    #     file_name = 'protocol_accuracy_repetition_%i.pdf' %repetition_to_plot
    # else:
    #     file_name = 'protocol_accuracy.pdf'
    # fig.savefig(os.path.join(path_to_save_figure, file_name), transparent = True)
    #
    # ### sm1 figure all group_sizes
    # fig, ax_arr = plt.subplots(2,3)
    # plt.subplots_adjust(left=.15, bottom=None, right=.85, top=None,
    #             wspace=.001, hspace=None)
    # plot_pair_of_statistics_per_group_sizes(fig, ax_arr,
    #                                     ratio_of_video_accumulated, overall_P2, group_sizes_list,
    #                                     scale_parameter_list, shape_parameter_list,
    #                                     'percentage of video accumulated', 'overal P2',
    #                                     [0,100], [0,100])
    # fig.set_size_inches((screen_x/100,screen_y/100))
    # set_protocol_accuracy_group_sizes(fig, ax_arr,
    #                                     'percentage of video accumulated', 'overal P2')
    # if repetition_to_plot is not None:
    #     file_name = 'percentage_accumulated_and_P2_repetition_%i.pdf' %repetition_to_plot
    # else:
    #     file_name = 'percentage_accumulated_and_P2.pdf'
    # fig.savefig(os.path.join(path_to_save_figure, file_name), transparent = True)
    #
    # ### sm2 figure all group_sizes
    # fig, ax_arr = plt.subplots(2,3)
    # plt.subplots_adjust(left=.15, bottom=None, right=.85, top=None,
    #             wspace=.001, hspace=None)
    # plot_pair_of_statistics_per_group_sizes(fig, ax_arr,
    #                                     accuracy_in_accumulation, accuracy_after_accumulation, group_sizes_list,
    #                                     scale_parameter_list, shape_parameter_list
    #                                     , 'accuracy in accumulation', 'accuracy in assignment',
    #                                     [0,100], [0,100])
    # fig.set_size_inches((screen_x/100,screen_y/100))
    # set_protocol_accuracy_group_sizes(fig, ax_arr,
    #                                     'accuracy in accumulation', 'accuracy in assignment')
    # if repetition_to_plot is not None:
    #     file_name = 'accuracy_before_and_in_accumulation_repetition_%i.pdf' %repetition_to_plot
    # else:
    #     file_name = 'accuracy_before_and_in_accumulation.pdf'
    # fig.savefig(os.path.join(path_to_save_figure, file_name), transparent = True)


    plt.show()
