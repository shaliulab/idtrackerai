from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('../')
sys.path.append('./utils')

import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, is_color_like
import matplotlib
MARKERS = matplotlib.markers.MarkerStyle.markers.keys()[5:]
import seaborn as sns
import pandas as pd
from pprint import pprint

from py_utils import get_spaced_colors_util

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_repetition_averaged_data_frame(results_data_frame):

    repetition_averaged_data_frame = pd.DataFrame(columns = [results_data_frame.mean().to_dict().keys() + ['individual_accuracies','individual_accuracies_95' , 'individual_accuracies_05']])
    count = 0
    for group_size in results_data_frame['group_size'].unique():

        for frames_in_video in results_data_frame['frames_in_video'].unique():

            for frames_in_fragment in results_data_frame['frames_per_fragment'].unique():

                temp_data_frame = results_data_frame.query('group_size == @group_size' +
                                                            ' & frames_in_video == @frames_in_video' +
                                                            ' & frames_per_fragment == @frames_in_fragment')
                temp_dict = temp_data_frame.mean().to_dict()
                repetition_averaged_data_frame.loc[count,:] = temp_dict

                individual_accuracies = []
                for repetition in results_data_frame['repetition'].unique():
                    results_data_frame_rep = results_data_frame.query('group_size == @group_size' +
                                                                ' & frames_in_video == @frames_in_video' +
                                                                ' & frames_per_fragment == @frames_in_fragment'+
                                                                ' & repetition == @repetition')

                    individual_accuracies.append(list(results_data_frame_rep['individual_accuracies_best']))
                individual_accuracies = list(flatten(individual_accuracies))
                repetition_averaged_data_frame.loc[count,'individual_accuracies'] = individual_accuracies
                repetition_averaged_data_frame.loc[count,'individual_accuracies_95'] = np.percentile(individual_accuracies,95)
                repetition_averaged_data_frame.loc[count,'individual_accuracies_05'] = np.percentile(individual_accuracies,5)

                count += 1
    return repetition_averaged_data_frame

def get_repetition_std_data_frame(results_data_frame):

    repetition_std_data_frame = pd.DataFrame(columns = [results_data_frame.std().to_dict().keys()])
    count = 0
    for group_size in results_data_frame['group_size'].unique():

        for frames_in_video in results_data_frame['frames_in_video'].unique():

            for frames_in_fragment in results_data_frame['frames_per_fragment'].unique():

                temp_data_frame = results_data_frame.query('group_size == @group_size' +
                                                            ' & frames_in_video == @frames_in_video' +
                                                            ' & frames_per_fragment == @frames_in_fragment')
                temp_dict = temp_data_frame.std().to_dict()
                repetition_std_data_frame.loc[count,:] = temp_dict

                count += 1
    return repetition_std_data_frame

if __name__ == '__main__':

    ### load global results data frame
    if os.path.isfile('./library/results_data_frame.pkl'):
        print("loading results_data_frame.pkl...")
        results_data_frame = pd.read_pickle('./library/library_tests_uncorrelated_different_networks/results_data_frame.pkl')
        print("results_data_frame.pkl loaded \n")
    else:
        print("results_data_frame.pkl does not exist \n")

    # get tests_data_frame and test to plot
    print("loading tests data frame")
    tests_data_frame = pd.read_pickle('./library/library_tests_uncorrelated_different_networks/tests_data_frame.pkl')
    test_names = [test_name for test_name in results_data_frame['test_name'].unique() if 'uncorrelated' in test_name]
    new_ordered_test_names = test_names[:-3]
    new_ordered_test_names.append(test_names[-2])
    new_ordered_test_names.append(test_names[-3])
    new_ordered_test_names.append(test_names[-1])

    cnn_model_names_dict = {0: 'idtracker.ai',
                            1: '1 conv layer',
                            2: '2 conv layers',
                            3: '4 conv layers',
                            6: 'fully 50',
                            7: 'fully 200',
                            8: 'fully 100 + fully 100',
                            9: 'fully 100 + fully 50',
                            10: 'fully 100 + fully 200'}

    legend_order_conv = ['idTracker',
                            cnn_model_names_dict[1],
                            cnn_model_names_dict[2],
                            '3 conv layers (idtracker.ai)',
                            cnn_model_names_dict[3]]

    legend_order_fully = ['idTracker',
                            cnn_model_names_dict[6],
                            'fully 100 (idtracker.ai)',
                            cnn_model_names_dict[7],
                            cnn_model_names_dict[9],
                            cnn_model_names_dict[8],
                            cnn_model_names_dict[10]]

    idTracker_single_image_accuracy_results = np.load('./library/library_tests_uncorrelated_different_networks/idTracker_results.npy')

    # plot
    plt.ion()
    sns.set_style("ticks", {'legend.frameon':True})
    fig1, ax_arr = plt.subplots(1,2, sharex = True)
    fig1.subplots_adjust(left=None, bottom=.15, right=None, top=.9,
                wspace=None, hspace=None)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig1.set_size_inches((screen_x*2/3/100,screen_y/1.75/100))
    fig1.canvas.set_window_title('Supplementary figure')
    # fig1.suptitle('Single image identification accuracy (MEAN) - libraries %s - %i repetitions' %('GHI',
    #                                                 len(results_data_frame['repetition'].unique())), fontsize = 25)

    fig2, ax2 = plt.subplots(1)
    fig2.set_size_inches((screen_x/3/100,screen_y/2.25/100))
    fig2.canvas.set_window_title('Main figure')
    fig2.subplots_adjust(left=.2, bottom=.15, right=None, top=.9,
                wspace=None, hspace=None)


    ax_arr[0].set_title('Convolutional modifications', fontsize = 20)
    ax_arr[1].set_title('Classification modifications', fontsize = 20)

    # import colorsys
    N = len(cnn_model_names_dict) - 1
    # HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    # RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    RGB_tuples = matplotlib.cm.get_cmap('jet')

    for i, test_name in enumerate(new_ordered_test_names[1:]):
        print("test name ", test_name)
        this_test_info = tests_data_frame[tests_data_frame['test_name'] == test_name]
        CNN_model = int(this_test_info.CNN_model)
        label = cnn_model_names_dict[CNN_model]
        results_data_frame_test = results_data_frame.query('test_name == @test_name')
        repetition_averaged_data_frame = get_repetition_averaged_data_frame(results_data_frame_test)
        repetition_std_data_frame = get_repetition_std_data_frame(results_data_frame_test)

        ''' accuracy '''
        repetition_averaged_data_frame = repetition_averaged_data_frame.apply(pd.to_numeric, errors='ignore')
        accuracy = repetition_averaged_data_frame.test_accuracy
        training_accuracy = repetition_averaged_data_frame.training_accuracy
        std_accuracy = repetition_std_data_frame.test_accuracy
        group_sizes = repetition_averaged_data_frame.group_size.astype('float32')

        if CNN_model <= 5:
            marker = MARKERS[i+1]
            color = RGB_tuples(i/N)
            ax = ax_arr[0]
            (_, caps, _) = ax.errorbar(group_sizes, accuracy * 100, yerr = std_accuracy * 100, color = color, label = label, marker = marker, capsize = 5)
            for cap in caps:
                cap.set_markeredgewidth(1)

        if CNN_model > 5:
            marker = MARKERS[i+1]
            color = RGB_tuples(i/N)
            ax = ax_arr[1]
            (_, caps, _) = ax.errorbar(group_sizes, accuracy * 100, yerr = std_accuracy * 100, color = color, label = label, marker = marker, capsize = 5)
            for cap in caps:
                cap.set_markeredgewidth(1)

    # idTracker accuracy
    ax_arr[0].plot(idTracker_single_image_accuracy_results * 100, 'k--', label = 'idTracker')
    ax_arr[1].plot(idTracker_single_image_accuracy_results * 100, 'k--', label = 'idTracker')
    ax2.plot(idTracker_single_image_accuracy_results * 100, 'k--', label = 'idTracker', linewidth = 3)

    # idTracker.ai accuracy
    test_name = test_names[0]
    results_data_frame_test = results_data_frame.query('test_name == @test_name')
    repetition_averaged_data_frame = get_repetition_averaged_data_frame(results_data_frame_test)
    repetition_std_data_frame = get_repetition_std_data_frame(results_data_frame_test)
    repetition_averaged_data_frame = repetition_averaged_data_frame.apply(pd.to_numeric, errors='ignore')
    accuracy = repetition_averaged_data_frame.test_accuracy
    std_accuracy = repetition_std_data_frame.test_accuracy
    group_sizes = repetition_averaged_data_frame.group_size.astype('float32')

    labels_to_plot = ['3 conv layers (idtracker.ai)', 'fully 100 (idtracker.ai)', 'idtracker.ai']
    axes_to_plot = [ax_arr[0], ax_arr[1], ax2]
    for ax, label in zip(axes_to_plot, labels_to_plot):
        if label == 'idtracker.ai':
            (_, caps, _) = ax.errorbar(group_sizes, accuracy * 100, yerr = std_accuracy * 100, color = 'k', label = label, marker = MARKERS[0], capsize = 5, zorder=3, linewidth = 3)
        else:
            (_, caps, _) = ax.errorbar(group_sizes, accuracy * 100, yerr = std_accuracy * 100, color = 'k', label = label, marker = MARKERS[0], capsize = 5, zorder=3)
        for cap in caps:
            cap.set_markeredgewidth(1)

    ax = ax_arr[0]
    ax.set_ylabel('accuracy',fontsize = 20)
    handles, labels = ax.get_legend_handles_labels()
    handles_ordered = []
    for label_ordered in legend_order_conv:
        index = labels.index(label_ordered)
        handles_ordered.append(handles[index])
    h_legend = ax.legend(handles_ordered, legend_order_conv, loc = 4)
    h_legend.set_title('CNN model')
    ax.set_xticks(results_data_frame.group_size.unique().astype(int))
    ax.set_xticklabels(results_data_frame.group_size.unique().astype(int))
    ax.set_xlabel('Group size', fontsize = 20)
    ax.set_ylabel('Single image accuracy (test)', fontsize = 20)
    ax.set_ylim([82,100])
    ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(ax = ax, right = True, top = True)

    ax = ax_arr[1]
    ax.set_yticklabels([])
    handles, labels = ax.get_legend_handles_labels()
    print('labels ', labels)
    handles_ordered = []
    for label_ordered in legend_order_fully:
        index = labels.index(label_ordered)
        handles_ordered.append(handles[index])
    h_legend = ax.legend(handles_ordered, legend_order_fully,loc = 4)
    h_legend.set_title('CNN model')
    ax.set_xticks(results_data_frame.group_size.unique().astype(int))
    ax.set_xticklabels(results_data_frame.group_size.unique().astype(int))
    ax.set_xlabel('Group size', fontsize = 20)
    ax.set_ylim([82,100])
    ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(ax = ax, right = True, top = True)

    h_legend = ax2.legend(loc = 4, prop={'size': 20})
    ax2.set_xticks([2, 10, 30, 60, 80, 100, 150])
    ax2.set_xticklabels([2, 10, 30, 60, 80, 100, 150])
    ax2.set_yticks([85, 90, 95, 100])
    ax2.set_yticklabels([85, 90, 95, 100])
    ax2.set_xlabel('Group size', fontsize = 20)
    ax2.set_ylabel('Single image accuracy (test)', fontsize = 20)
    ax2.set_ylim([82,100])
    ax2.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax2.tick_params(axis='both', which='major', labelsize=20)
    sns.despine(ax = ax2, right = True, top = True)

    plt.minorticks_off()

    # plt.show()
    fig1.savefig('./library/library_tests_uncorrelated_different_networks/single_image_identification_accuracy_SM.pdf', transparent=True)
    fig2.savefig('./library/library_tests_uncorrelated_different_networks/single_image_identification_accuracy.pdf', transparent=True)
