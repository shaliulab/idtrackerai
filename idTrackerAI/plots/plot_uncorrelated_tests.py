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
        results_data_frame = pd.read_pickle('./library/results_data_frame_0.pkl')
        print("results_data_frame.pkl loaded \n")
    else:
        print("results_data_frame.pkl does not exist \n")

    # get tests_data_frame and test to plot
    print("loading tests data frame")
    tests_data_frame = pd.read_pickle('./library/tests_data_frame.pkl')
    test_names = [test_name for test_name in results_data_frame['test_name'].unique() if 'uncorrelated' in test_name]
    print("test_names: ", test_names)
    cnn_model_names_dict = {0: 'our network',
                            1: '1 conv layer',
                            2: '2 conv layers',
                            3: '4 conv layers',
                            4: 'inverted network',
                            5: 'no ReLu network',
                            6: 'fully 50',
                            7: 'fully 200',
                            8: 'fully 100 + fully 100',
                            9: 'fully 100 + fully 50',
                            10: 'fully 100 + fully 200'}

    # plot
    plt.ion()
    sns.set_style("ticks")
    fig, ax_arr = plt.subplots(2,2, sharex = True)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x*2/3/100,screen_y/100))
    fig.suptitle('Single image identification accuracy - library %s - %i repetitions' %('G',
                                                    len(results_data_frame['repetition'].unique())), fontsize = 25)

    ax_arr[0,0].set_title('Convolutional modifications', fontsize = 20)
    ax_arr[0,1].set_title('Classification modifications', fontsize = 20)

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

        CNN_model = int(this_test_info.CNN_model)
        label = cnn_model_names_dict[CNN_model]

        ''' accuracy '''
        repetition_averaged_data_frame = repetition_averaged_data_frame.apply(pd.to_numeric, errors='ignore')
        accuracy = repetition_averaged_data_frame.accuracy
        per95_accuracy = repetition_averaged_data_frame.individual_accuracies_95
        per05_accuracy = repetition_averaged_data_frame.individual_accuracies_05
        group_sizes = repetition_averaged_data_frame.group_size.astype('float32')

        if CNN_model == 0 or CNN_model <= 5:
            ax = ax_arr[0,0]
            ax.plot(group_sizes, accuracy, color = RGB_tuples(CNN_model/N),label = label, marker = MARKERS[CNN_model])
            ax.set_ylabel('accuracy',fontsize = 20)
            h_legend = ax.legend()
            h_legend.set_title('CNN model')
        if CNN_model == 0 or CNN_model > 5:
            ax = ax_arr[0,1]
            ax.plot(group_sizes, accuracy, color = RGB_tuples(CNN_model/N),label = label, marker = MARKERS[CNN_model])
            ax.set_yticklabels([])
            h_legend = ax.legend()
            h_legend.set_title('CNN model')
        ax.set_xticklabels(results_data_frame.group_size.unique().astype(int))
        ax.set_ylim([0.9,1.])
        ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
        ax.tick_params(axis='both', which='major', labelsize=16)

        ''' time '''
        if CNN_model == 0 or CNN_model <= 5:
            ax = ax_arr[1,0]
            repetition_averaged_data_frame.plot(x = 'group_size',
                                                y = 'total_time',
                                                c = np.asarray(RGB_tuples(CNN_model/N)),
                                                ax = ax,
                                                marker = MARKERS[CNN_model])
            ax.set_ylabel('training time (sec)', fontsize = 20)
            ax.legend_.remove()
            ax.set_xlabel('group size', fontsize = 20)
        if CNN_model == 0 or CNN_model > 5:
            ax = ax_arr[1,1]
            repetition_averaged_data_frame.plot(x = 'group_size',
                                                y = 'total_time',
                                                c = np.asarray(RGB_tuples(CNN_model/N)),
                                                ax = ax,
                                                marker = MARKERS[CNN_model])

            ax.legend_.remove()
            ax.set_yticklabels([])
            ax.set_xlabel('group size', fontsize = 20)
        ax.set_xticks(list(this_test_info.group_sizes)[0])
        ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
        ax.set_ylim([0,2000])
        ax.tick_params(axis='both', which='major', labelsize=16)


    plt.minorticks_off()

    # plt.show()
    fig.savefig('single_image_identification_accuracy.pdf', transparent=True)
