from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('../')
sys.path.append('./utils')

import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, is_color_like
import seaborn as sns
import pandas as pd
from pprint import pprint

from idtrackerai.utils.py_utils import  get_spaced_colors_util

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_repetition_averaged_data_frame(results_data_frame):

    repetition_averaged_data_frame = pd.DataFrame(columns = [results_data_frame.median().to_dict().keys() + ['individual_accuracies','individual_accuracies_95' , 'individual_accuracies_05']])
    count = 0
    for group_size in results_data_frame['group_size'].unique():

        for frames_in_video in results_data_frame['frames_in_video'].unique():

            for frames_in_fragment in results_data_frame['frames_per_fragment'].unique():

                temp_data_frame = results_data_frame.query('group_size == @group_size' +
                                                            ' & frames_in_video == @frames_in_video' +
                                                            ' & frames_per_fragment == @frames_in_fragment')
                temp_dict = temp_data_frame.median().to_dict()
                repetition_averaged_data_frame.loc[count,:] = temp_dict

                individual_accuracies = []
                for repetition in results_data_frame['repetition'].unique():
                    results_data_frame_rep = results_data_frame.query('group_size == @group_size' +
                                                                ' & frames_in_video == @frames_in_video' +
                                                                ' & frames_per_fragment == @frames_in_fragment'+
                                                                ' & repetition == @repetition')

                    individual_accuracies.append(list(results_data_frame_rep['individual_accuracies']))
                individual_accuracies = list(flatten(individual_accuracies))
                repetition_averaged_data_frame.loc[count,'individual_accuracies'] = individual_accuracies
                repetition_averaged_data_frame.loc[count,'individual_accuracies_95'] = np.percentile(individual_accuracies,95)
                repetition_averaged_data_frame.loc[count,'individual_accuracies_05'] = np.percentile(individual_accuracies,5)

                count += 1
    return repetition_averaged_data_frame

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
    test_names = [test_name for test_name in results_data_frame['test_name'].unique() if 'trainonly1GF' in test_name]
    print("test_names: ", test_names)
    cnn_model_names_dict = {0: 'our network',
                            1: '1 conv layer',
                            2: '2 conv layers',
                            3: '4 conv layers',
                            4: 'inverted network',
                            5: 'no ReLu network'}

    # plot
    plt.ion()
    fig, ax_arr = plt.subplots(2,2)
    fig.suptitle('%s - library %s - %i repetitions' %('Uncorrelated images test',
                                                    'DEF',
                                                    len(results_data_frame['repetition'].unique())))
    ax_arr[0,0].set_title('Portrait preprocessing')
    ax_arr[0,1].set_title('PCA preprocessing')

    import colorsys
    N = len(cnn_model_names_dict)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    for test_name in test_names[::-1]:
        this_test_info = tests_data_frame[tests_data_frame['test_name'] == test_name]
        # get results from the test to plot
        results_data_frame_test = results_data_frame.query('test_name == @test_name')

        # average the repetitions
        repetition_averaged_data_frame = get_repetition_averaged_data_frame(results_data_frame_test)

        if 'DEF' in test_name:
            column = 0
        elif 'GHI' in test_name:
            column = 1
        CNN_model = int(this_test_info.CNN_model)
        label = cnn_model_names_dict[CNN_model]

        ''' accuracy '''
        ax = ax_arr[0,column]
        repetition_averaged_data_frame = repetition_averaged_data_frame.apply(pd.to_numeric, errors='ignore')
        accuracy = repetition_averaged_data_frame.accuracy
        per95_accuracy = repetition_averaged_data_frame.individual_accuracies_95
        per05_accuracy = repetition_averaged_data_frame.individual_accuracies_05
        group_sizes = repetition_averaged_data_frame.group_size.astype('float32')

        ax.plot(group_sizes, accuracy, color=np.asarray(RGB_tuples[CNN_model]),label = label)
        # ax.fill_between(group_sizes,per95_accuracy,per05_accuracy,
        #                 alpha = .1,
        #                 facecolor = np.asarray(RGB_tuples[CNN_model]),
        #                 edgecolor = np.asarray(RGB_tuples[CNN_model]),
        #                 linewidth = 1)

        ax.set_ylabel('accuracy')
        h_legend = ax.legend()
        h_legend.set_title('CNN_model')
        ax.set_xticks(list(this_test_info.group_sizes)[0])
        ax.set_ylim([0.7,1.])
        ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])

        ''' time '''
        ax = ax_arr[1,column]
        repetition_averaged_data_frame.plot(x = 'group_size',
                                            y = 'total_time',
                                            c = np.asarray(RGB_tuples[CNN_model]),
                                            ax = ax)
        ax.set_ylabel('total_time (sec)')
        ax.legend_.remove()
        ax.set_xticks(list(this_test_info.group_sizes)[0])
        ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
        ax.set_ylim([0,3500])

    plt.show()
