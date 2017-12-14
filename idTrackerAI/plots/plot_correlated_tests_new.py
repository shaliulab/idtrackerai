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
    scale_parameter_list = test_dictionary['scale_parameter'][::-1]
    shape_parameter_list = test_dictionary['shape_parameter'][::-1]
    number_of_conditions_mean = len(test_dictionary['scale_parameter'])
    number_of_condition_var = len(test_dictionary['shape_parameter'])
    number_of_repetitions = len(results_data_frame.repetition.unique())
    protocol = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    total_time = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    ratio_of_accumulated_images = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    ratio_of_video_accumulated = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    overall_P2 = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    accuracy = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    accuracy_in_accumulation = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    accuracy_after_accumulation = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    th_mean = np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))
    th_std =np.zeros((number_of_condition_var, number_of_conditions_mean, number_of_repetitions))

    plt.ion()
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    sns.set_style("ticks")
    fig_distributions_list = []
    fig_mean_std_list = []
    fig_statistics_list = []
    for group_size in results_data_frame.group_size.unique():

        for frames_in_video in results_data_frame.frames_in_video.unique():
            fig_distributions, ax_arr = plt.subplots(len(results_data_frame.loc[:,'scale_parameter'].unique()), len(results_data_frame.loc[:,'shape_parameter'].unique()),
                                        sharex = True, sharey = True)
            fig_distributions_list.append(fig_distributions)
            fig_distributions.suptitle('Group size %i - Frames in video %i' %(group_size, frames_in_video))


            for i, scale_parameter in enumerate(scale_parameter_list):
                if scale_parameter % 1 == 0:
                    scale_parameter = int(scale_parameter)

                for j, shape_parameter in enumerate(shape_parameter_list):
                    print('----- ', scale_parameter, shape_parameter)
                    if shape_parameter % 1 == 0:
                        shape_parameter = int(shape_parameter)

                    for k, repetition in enumerate(results_data_frame.repetition.unique()):
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
                        ### Plot distributions
                        if repetition == 1:
                            nbins = 10
                            number_of_images_in_individual_fragments = results_data_frame_rep['individual_fragments_lengths'].item()
                            # number_of_images_in_individual_fragments = number_of_images_in_individual_fragments[number_of_images_in_individual_fragments >= 3]
                            gamma_simulation = gamma(shape_parameter, loc = 0.99, scale = scale_parameter)
                            gamma_simulation_logpdf = pdf2logpdf(gamma_simulation.pdf)
                            ax = ax_arr[j,i]
                            MIN = np.min(number_of_images_in_individual_fragments)
                            MAX = np.max(number_of_images_in_individual_fragments)
                            logbins = np.linspace(np.log10(MIN), np.log10(MAX), nbins)
                            ax.hist(np.log10(number_of_images_in_individual_fragments), bins = logbins, normed = True)
                            logbins_pdf = np.linspace(np.log10(MIN), np.log10(MAX), 100)
                            ax.plot(logbins_pdf, gamma_simulation_logpdf(np.power(10,logbins_pdf)))
                            if j == len(shape_parameter_list)-1:
                                ax.set_xlabel('number of frames \n\nscale = %.2f' %scale_parameter)
                            if i == 0:
                                ax.set_ylabel('shape = %.2f \n\nPDF' %shape_parameter)
                            mean = shape_parameter * scale_parameter
                            sigma = np.sqrt(shape_parameter * scale_parameter**2)
                            title = r'$\mu$ = %.2f, $\sigma$ = %.2f' %(mean, sigma)
                            ax.set_xlim((np.log10(MIN), np.log10(MAX)))
                            ax.set_xticks([1,2,3])
                            ax.set_xticklabels([10,100,1000])
                            ax.text(2.25, 1.15, title, horizontalalignment = 'center')
                            ax.set_ylim((0,1.3))

                        if len(results_data_frame_rep) != 0:
                            protocol[j,i,k] = results_data_frame_rep.protocol.item() if video_object_found else None
                            total_time[j,i,k] = results_data_frame_rep.total_time.item()  if video_object_found else None
                            ratio_of_accumulated_images[j,i,k] = (results_data_frame_rep.number_of_partially_accumulated_individual_blobs
                                                                    + results_data_frame_rep.number_of_globally_accumulated_individual_blobs) / \
                                                                    (video.individual_fragments_stats['number_of_accumulable_individual_blobs'])  if video_object_found else None
                            ratio_of_video_accumulated[j,i,k] = (results_data_frame_rep.number_of_partially_accumulated_individual_blobs
                                                                    + results_data_frame_rep.number_of_globally_accumulated_individual_blobs) / \
                                                                    results_data_frame_rep.number_of_blobs if video_object_found else None

                            overall_P2[j,i,k] = video.overall_P2 if video_object_found else None
                            accuracy[j,i,k] = results_data_frame_rep.accuracy.item() if video_object_found else None
                            accuracy_in_accumulation[j,i,k] = results_data_frame_rep.accuracy_in_accumulation.item() if video_object_found else None
                            accuracy_after_accumulation[j,i,k] = results_data_frame_rep.accuracy_after_accumulation.item() if video_object_found else None

                            th_mean[j,i,k] = np.mean(number_of_images_in_individual_fragments)
                            th_std[j,i,k] = np.std(number_of_images_in_individual_fragments)

            fig_statistics, ax_arr = plt.subplots(2,4)
            fig_statistics_list.append(fig_statistics)
            fig_statistics.suptitle('Group size %i - Frames in video %i' %(group_size, frames_in_video))

            def plot_statistics_heatmap(ax, matrix, title, xticklabels, yticklabels, vmax = None, vmin = None, annot = True):
                if title == 'total time':
                    fmt = '.1f'
                elif title == 'protocol':
                    fmt = '.0f'
                else:
                    fmt = '.4f'
                ax = sns.heatmap(np.mean(matrix, axis = 2),
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

            plot_statistics_heatmap(ax_arr[0,0], protocol, 'protocol', scale_parameter_list, shape_parameter_list)
            plot_statistics_heatmap(ax_arr[0,1], total_time, 'total time', scale_parameter_list, shape_parameter_list)
            plot_statistics_heatmap(ax_arr[0,2], ratio_of_accumulated_images, r'$\%$' + ' accumulated images', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0 )
            plot_statistics_heatmap(ax_arr[0,3], ratio_of_video_accumulated, r'$\%$' + ' video', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)
            plot_statistics_heatmap(ax_arr[1,0], overall_P2, 'overall P2', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)
            plot_statistics_heatmap(ax_arr[1,1], accuracy, 'accuracy', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)
            plot_statistics_heatmap(ax_arr[1,2], accuracy_in_accumulation, 'accuracy in accumulation', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)
            plot_statistics_heatmap(ax_arr[1,3], accuracy_after_accumulation, 'accuracy after accumulation', scale_parameter_list, shape_parameter_list, vmax = 1, vmin = 0)

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

            fig_mean_std, ax_arr2 = plt.subplots(1,2)
            fig_mean_std_list.append(fig_mean_std)
            fig_mean_std.suptitle('Group size %i - Frames in video %i' %(group_size, frames_in_video))

            plot_statistics_heatmap(ax_arr2[0], th_mean, 'mean', scale_parameter_list, shape_parameter_list)
            plot_statistics_heatmap(ax_arr2[1], th_std, 'std', scale_parameter_list, shape_parameter_list)


    path_to_save_figure = os.path.join('./library','library_test_' + results_data_frame.test_name.unique()[0],
                                    'group_size_' + str(int(group_size)))



    [fig.set_size_inches((screen_x/100,screen_y/100)) for fig in fig_statistics_list + fig_distributions_list + fig_mean_std_list]
    [fig.savefig(os.path.join(path_to_save_figure, 'distributions_%i.pdf' %video_length), transparent = True) for video_length, fig in zip(results_data_frame.frames_in_video.unique(), fig_distributions_list)]
    [fig.savefig(os.path.join(path_to_save_figure, 'statistics_%i.pdf' %video_length), transparent = True) for video_length, fig in zip(results_data_frame.frames_in_video.unique(), fig_statistics_list)]
    [fig.savefig(os.path.join(path_to_save_figure, 'mean_std_%i.pdf' %video_length), transparent = True) for video_length, fig in zip(results_data_frame.frames_in_video.unique(), fig_mean_std_list)]


    plt.show()
