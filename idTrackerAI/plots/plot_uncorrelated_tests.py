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

                # individual_accuracies = []
                # for repetition in results_data_frame['repetition'].unique():
                #     results_data_frame_rep = results_data_frame.query('group_size == @group_size' +
                #                                                 ' & frames_in_video == @frames_in_video' +
                #                                                 ' & frames_per_fragment == @frames_in_fragment'+
                #                                                 ' & repetition == @repetition')
                #
                #     individual_accuracies.append(list(results_data_frame_rep['individual_accuracies_best']))
                # individual_accuracies = list(flatten(individual_accuracies))
                # repetition_std_data_frame.loc[count,'individual_accuracies'] = individual_accuracies
                # repetition_std_data_frame.loc[count,'individual_accuracies_95'] = np.percentile(individual_accuracies,95)
                # repetition_std_data_frame.loc[count,'individual_accuracies_05'] = np.percentile(individual_accuracies,5)

                count += 1
    return repetition_std_data_frame

if __name__ == '__main__':

    ### load global results data frame
    if os.path.isfile('./library/results_data_frame.pkl'):
        print("loading results_data_frame.pkl...")
        results_data_frame = pd.read_pickle('./library/results_data_frame.pkl')
        print("results_data_frame.pkl loaded \n")
    else:
        print("results_data_frame.pkl does not exist \n")

    # get tests_data_frame and test to plot
    print("loading tests data frame")
    tests_data_frame = pd.read_pickle('./library/tests_data_frame.pkl')
    test_names = [test_name for test_name in results_data_frame['test_name'].unique() if 'uncorrelated' in test_name]
    print("test_names: ", test_names)
    cnn_model_names_dict = {0: 'idTracker.ai network',
                            1: '1 conv layer',
                            2: '2 conv layers',
                            3: '4 conv layers',
                            4: 'inverted network',
                            5: 'linear network but max-pooling',
                            6: 'fully 50',
                            7: 'fully 200',
                            8: 'fully 100 + fully 100',
                            9: 'fully 100 + fully 50',
                            10: 'fully 100 + fully 200'}

    idTracker_single_image_accuracy_results = np.load('/home/chronos/Desktop/IdTrackerDeep/idTrackerAI/library/idTracker_results.npy')
    print(idTracker_single_image_accuracy_results)

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
    fig2.canvas.set_window_title('Main figure')
    fig2.subplots_adjust(left=.2, bottom=.15, right=None, top=.9,
                wspace=None, hspace=None)


    ax_arr[0].set_title('Convolutional modifications', fontsize = 20)
    ax_arr[1].set_title('Classification modifications', fontsize = 20)

    ax_arr[0].plot(idTracker_single_image_accuracy_results, 'k-', label = 'idTracker')
    ax_arr[1].plot(idTracker_single_image_accuracy_results, 'k-', label = 'idTracker')
    ax2.plot(idTracker_single_image_accuracy_results, 'k-', label = 'idTracker')

    # import colorsys
    N = len(cnn_model_names_dict)
    # HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    # RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    RGB_tuples = matplotlib.cm.get_cmap('jet')

    for test_name in test_names[::-1]:
        this_test_info = tests_data_frame[tests_data_frame['test_name'] == test_name]
        # get results from the test to plot
        results_data_frame_test = results_data_frame.query('test_name == @test_name')

        # average the repetitions
        repetition_averaged_data_frame = get_repetition_averaged_data_frame(results_data_frame_test)
        repetition_std_data_frame = get_repetition_std_data_frame(results_data_frame_test)

        CNN_model = int(this_test_info.CNN_model)
        label = cnn_model_names_dict[CNN_model]

        ''' accuracy '''
        repetition_averaged_data_frame = repetition_averaged_data_frame.apply(pd.to_numeric, errors='ignore')
        accuracy = repetition_averaged_data_frame.accuracy_best
        std_accuracy = repetition_std_data_frame.accuracy_best
        per95_accuracy = repetition_averaged_data_frame.individual_accuracies_95
        per05_accuracy = repetition_averaged_data_frame.individual_accuracies_05
        group_sizes = repetition_averaged_data_frame.group_size.astype('float32')

        if CNN_model == 0 or CNN_model <= 5:
            ax = ax_arr[0]
            (_, caps, _) = ax.errorbar(group_sizes, accuracy, yerr=std_accuracy, color = RGB_tuples(CNN_model/N), label = label, marker = MARKERS[CNN_model], capsize = 5)
            for cap in caps:
                cap.set_markeredgewidth(1)

            if CNN_model == 0:
                (_, caps, _) = ax2.errorbar(group_sizes, accuracy, yerr=std_accuracy, color = RGB_tuples(CNN_model/N), label = label, marker = MARKERS[CNN_model], capsize = 5)
                for cap in caps:
                    cap.set_markeredgewidth(1)

        if CNN_model == 0 or CNN_model > 5:
            ax = ax_arr[1]
            (_, caps, _) = ax.errorbar(group_sizes, accuracy, yerr=std_accuracy, color = RGB_tuples(CNN_model/N),label = label, marker = MARKERS[CNN_model], capsize = 5)
            for cap in caps:
                cap.set_markeredgewidth(1)

    ax = ax_arr[0]
    ax.set_ylabel('accuracy',fontsize = 20)
    h_legend = ax.legend(loc = 4)
    h_legend.set_title('CNN model')
    ax.set_xticks(results_data_frame.group_size.unique().astype(int))
    ax.set_xticklabels(results_data_frame.group_size.unique().astype(int))
    ax.set_xlabel('Group size', fontsize = 20)
    ax.set_ylabel('Accuracy', fontsize = 20)
    ax.set_ylim([0.75,1.])
    ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax = ax_arr[1]
    ax.set_yticklabels([])
    h_legend = ax.legend(loc = 4)
    h_legend.set_title('CNN model')
    ax.set_xticks(results_data_frame.group_size.unique().astype(int))
    ax.set_xticklabels(results_data_frame.group_size.unique().astype(int))
    ax.set_xlabel('Group size', fontsize = 20)
    ax.set_ylim([0.75,1.])
    ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax.tick_params(axis='both', which='major', labelsize=16)

    h_legend = ax2.legend(loc = 4)
    h_legend.set_title('CNN model')
    ax2.set_xticks(results_data_frame.group_size.unique().astype(int))
    ax2.set_xticklabels(results_data_frame.group_size.unique().astype(int))
    ax2.set_xlabel('Group size', fontsize = 20)
    ax2.set_ylabel('Accuracy', fontsize = 20)
    ax2.set_ylim([0.82,1.])
    ax2.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax2.tick_params(axis='both', which='major', labelsize=16)

    plt.minorticks_off()

    # plt.show()
    fig1.savefig('single_image_identification_accuracy_SM.pdf', transparent=True)
    fig2.savefig('single_image_identification_accuracy.pdf', transparent=True)
