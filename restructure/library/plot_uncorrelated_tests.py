from __future__ import absolute_import, division, print_function
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pprint import pprint

def get_repetition_averaged_data_frame(results_data_frame):

    repetition_averaged_data_frame = pd.DataFrame(columns = [results_data_frame.mean().to_dict().keys()])
    count = 0
    for group_size in results_data_frame['group_size'].unique():

        for frames_in_video in results_data_frame['frames_in_video'].unique():

            for frames_in_fragment in results_data_frame['frames_per_fragment'].unique():

                temp_data_frame = results_data_frame.query('group_size == @group_size' +
                                                            ' & frames_in_video == @frames_in_video' +
                                                            ' & frames_per_fragment == @frames_in_fragment')
                temp_dict = temp_data_frame.mean().to_dict()
                repetition_averaged_data_frame.loc[count,:] = temp_dict

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

    # plot
    plt.ion()
    fig, ax_arr = plt.subplots(2,1)

    ### Accuracy
    fig.suptitle('%s - library %s - %i repetitions' %('Uncorrelated images test',
                                                    results_data_frame.loc[0,'IMDB_codes'],
                                                    len(results_data_frame['repetition'].unique())))

    test_names = [test_name for test_name in results_data_frame['test_name'].unique() if 'uncorrelated' in test_name and 'DEF' in test_name]
    print("test_names: ", test_names)
    cnn_model_names_dict = {0: 'our network',
                            1: '1 conv layer',
                            2: '2 conv layers'}

    for test_name in test_names:
        # get results from the test to plot
        results_data_frame_test = results_data_frame.query('test_name == @test_name')

        # average the repetitions
        repetition_averaged_data_frame = get_repetition_averaged_data_frame(results_data_frame_test)

        repetition_averaged_data_frame.plot(x='group_size',
                                            y = 'accuracy',
                                            ax = ax_arr[0],
                                            label = cnn_model_names_dict[repetition_averaged_data_frame.loc[0,'CNN_model']])
        ax_arr[0].set_ylabel('accuracy')
        h_legend = ax_arr[0].legend()
        h_legend.set_title('CNN_model')
        ax_arr[0].set_ylim([0.9,1.])
        ax_arr[0].set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])


        repetition_averaged_data_frame.plot(x = 'group_size',
                                            y = 'total_time',
                                            ax = ax_arr[1])
        ax_arr[1].set_ylabel('total_time (sec)')
        ax_arr[1].legend_.remove()
        ax_arr[1].set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    plt.show()
