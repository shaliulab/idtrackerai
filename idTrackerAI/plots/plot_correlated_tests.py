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
from scipy.stats import gamma
from scipy import stats

from py_utils import get_spaced_colors_util
from library_utils import LibraryJobConfig

def pdf2logpdf(pdf):
    def logpdf(x):
        return pdf(x)*x*np.log(10)
    return logpdf

def plot_statistics_heatmap(ax, matrix, title, xticklabels, yticklabels, vmax = None, vmin = None, annot = True):
    if title == 'total time':
        fmt = '.1f'
    elif title == 'protocol-main' or title == 'accuracy-main':
        fmt = 's'
    elif title == 'protocol':
        fmt = '.0f'
    else:
        fmt = '.4f'
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
    protocol = np.nanmedian(protocol, axis = 3)
    total_time = np.nanmean(total_time, axis = 3)
    ratio_of_accumulated_images = np.nanmean(ratio_of_accumulated_images, axis = 3)
    ratio_of_video_accumulated = np.nanmean(ratio_of_video_accumulated, axis = 3)
    overall_P2 = np.nanmean(overall_P2, axis = 3)
    accuracy = np.nanmean(accuracy, axis = 3)
    accuracy_in_accumulation = np.nanmean(accuracy_in_accumulation, axis = 3)
    accuracy_after_accumulation = np.nanmean(accuracy_after_accumulation, axis = 3)

    plot_statistics_heatmap(ax_arr[0,0], protocol[i], 'protocol', scale_parameter_list, shape_parameter_list, vmin = 1, vmax = 3)
    plot_statistics_heatmap(ax_arr[0,1], total_time[i], 'total time', scale_parameter_list, shape_parameter_list)
    plot_statistics_heatmap(ax_arr[0,2], ratio_of_accumulated_images[i], r'$\%$' + ' accumulated images', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0 )
    plot_statistics_heatmap(ax_arr[0,3], ratio_of_video_accumulated[i], r'$\%$' + ' video', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)
    plot_statistics_heatmap(ax_arr[1,0], overall_P2[i], 'overall P2', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)
    plot_statistics_heatmap(ax_arr[1,1], accuracy[i], 'accuracy', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)
    plot_statistics_heatmap(ax_arr[1,2], accuracy_in_accumulation[i], 'accuracy in accumulation', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)
    plot_statistics_heatmap(ax_arr[1,3], accuracy_after_accumulation[i], 'accuracy after accumulation', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)

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
                if ratio_of_accumulated_images[group_size, row, column] >= .9:
                    protocol_annotate[group_size, row, column] = '%.0f' %protocol[group_size, row, column]
                    accuracy_annotate[group_size, row, column] = '%.4f' %accuracy[group_size, row, column]
                else:
                    protocol_annotate[group_size, row, column] = '%.0f \n(%.1f)' %(protocol[group_size, row, column], 100*ratio_of_video_accumulated[group_size, row, column])
                    accuracy_annotate[group_size, row, column] = '%.3f' %accuracy_in_accumulation[group_size, row, column]
                    # accuracy_annotate[group_size, row, column] = '%.4f \n(%.3f)' %(accuracy[group_size, row, column], accuracy_in_accumulation[group_size, row, column])
                    accuracy[group_size, row, column] = accuracy_in_accumulation[group_size, row, column]


    return protocol, accuracy, protocol_annotate, accuracy_annotate

def plot_protocol_accuracy_group_sizes(fig_protocol_accuracy, ax_arr, stats1, stats2,
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
    ax.text(2.25, 1.15, th_title, horizontalalignment = 'center')
    ax.text(2.25, 1.0, title, horizontalalignment = 'center')
    ax.set_ylim((0,1.3))
    if shape_parameter == 0.05 and scale_parameter == 100:
        ax.legend(loc = 7)

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

    plt.ion()
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    # sns.set_style("white")
    for i, group_size in enumerate(group_sizes_list):
        print("***group_size ", group_size)
        fig_distributions, ax_arr = plt.subplots(len(results_data_frame.loc[:,'scale_parameter'].unique()), len(results_data_frame.loc[:,'shape_parameter'].unique()),
                                    sharex = True, sharey = True)
        fig_distributions.suptitle('Group size %i' %group_size)
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

                    ### Create accuracy matrix
                    results_data_frame_rep = results_data_frame.query('group_size == @group_size' +
                                                                ' & frames_in_video == @frames_in_video' +
                                                                ' & scale_parameter == @scale_parameter' +
                                                                ' & shape_parameter == @shape_parameter' +
                                                                ' & repetition == @repetition')
                    if results_data_frame_rep.protocol.item() is None:
                        video_object_found = False
                        print("Algorithm did not finish")
                    ### Plot distributions
                    if repetition == 1:
                        try:
                            nbins = 25
                            number_of_images_in_individual_fragments = results_data_frame_rep['individual_fragments_lengths'].item()
                            ax = ax_arr[k,j]
                            plot_histogram_individual_fragments(ax, number_of_images_in_individual_fragments, scale_parameter, shape_parameter)
                        except:
                            number_of_images_in_individual_fragments = np.nan

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

        # All statistics figure
        fig_statistics, ax_arr = plt.subplots(2,4)
        fig_statistics.suptitle('Group size %i' %group_size)
        plot_all_statistics_figure(ax_arr, protocol, total_time, ratio_of_accumulated_images,
                                    ratio_of_video_accumulated, overall_P2,
                                    accuracy, accuracy_in_accumulation,
                                    accuracy_after_accumulation,
                                    scale_parameter_list, shape_parameter_list)
        set_properties_all_statistics_figure(ax_arr)

        path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0],
                                        'group_size_' + str(int(group_size)))
        print(path_to_save_figure)
        [fig.set_size_inches((screen_x/100,screen_y/100)) for fig in [fig_distributions, fig_statistics]]
        fig_distributions.savefig(os.path.join(path_to_save_figure, 'distributions_%i.pdf' %group_size), transparent = True)
        fig_statistics.savefig(os.path.join(path_to_save_figure, 'statistics_%i.pdf' %group_size), transparent = True)


    ### Compute average over repetitions
    protocol = np.median(protocol, axis = 3)
    total_time = np.mean(total_time, axis = 3)
    ratio_of_accumulated_images = np.mean(ratio_of_accumulated_images, axis = 3)
    ratio_of_video_accumulated = np.mean(ratio_of_video_accumulated, axis = 3)
    overall_P2 = np.mean(overall_P2, axis = 3)
    accuracy = np.mean(accuracy, axis = 3)
    accuracy_in_accumulation = np.mean(accuracy_in_accumulation, axis = 3)
    accuracy_after_accumulation = np.mean(accuracy_after_accumulation, axis = 3)

    ### main figure all group_sizes
    # build annotate matrices
    protocol, accuracy, \
    protocol_annotate, accuracy_annotate = build_annotate_matrices(protocol, accuracy,
                                                                accuracy_in_accumulation,
                                                                ratio_of_video_accumulated,
                                                                ratio_of_accumulated_images)
    fig_protocol_accuracy, ax_arr = plt.subplots(2,3)
    plt.subplots_adjust(left=.15, bottom=None, right=.85, top=None,
                wspace=.001, hspace=None)
    plot_protocol_accuracy_group_sizes(fig_protocol_accuracy, ax_arr,
                                        protocol, accuracy, group_sizes_list,
                                        scale_parameter_list, shape_parameter_list,
                                        'protocol-main', 'accuracy-main',
                                        [1,3], [0,1],
                                        protocol_annotate, accuracy_annotate)
    fig_protocol_accuracy.set_size_inches((screen_x/100,screen_y/100))
    set_protocol_accuracy_group_sizes(fig_protocol_accuracy, ax_arr,
                                    'protocol', 'accuracy')
    path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0])
    fig_protocol_accuracy.savefig(os.path.join(path_to_save_figure, 'protocol_accuracy.pdf'), transparent = True)

    ### sm1 figure all group_sizes
    fig_protocol_accuracy, ax_arr = plt.subplots(2,3)
    plt.subplots_adjust(left=.15, bottom=None, right=.85, top=None,
                wspace=.001, hspace=None)
    plot_protocol_accuracy_group_sizes(fig_protocol_accuracy, ax_arr,
                                        ratio_of_video_accumulated, overall_P2, group_sizes_list,
                                        scale_parameter_list, shape_parameter_list,
                                        'percentage of video accumulated', 'overal P2',
                                        [0,1], [0,1])
    fig_protocol_accuracy.set_size_inches((screen_x/100,screen_y/100))
    set_protocol_accuracy_group_sizes(fig_protocol_accuracy, ax_arr,
                                        'percentage of video accumulated', 'overal P2')
    path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0])
    fig_protocol_accuracy.savefig(os.path.join(path_to_save_figure, 'percentage_accumulated_and_P2.pdf'), transparent = True)

    ### sm2 figure all group_sizes
    fig_protocol_accuracy, ax_arr = plt.subplots(2,3)
    plt.subplots_adjust(left=.15, bottom=None, right=.85, top=None,
                wspace=.001, hspace=None)
    plot_protocol_accuracy_group_sizes(fig_protocol_accuracy, ax_arr,
                                        accuracy_in_accumulation, accuracy_after_accumulation, group_sizes_list,
                                        scale_parameter_list, shape_parameter_list
                                        , 'accuracy in accumulation', 'accuracy after accumulation',
                                        [0,1], [0,1])
    fig_protocol_accuracy.set_size_inches((screen_x/100,screen_y/100))
    set_protocol_accuracy_group_sizes(fig_protocol_accuracy, ax_arr,
                                        'accuracy in accumulation', 'accuracy after accumulation')
    path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0])
    fig_protocol_accuracy.savefig(os.path.join(path_to_save_figure, 'accuracy_before_and_in_accumulation.pdf'), transparent = True)


    plt.show()
