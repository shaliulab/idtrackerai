from __future__ import absolute_import, division, print_function
# Import standard libraries
import sys

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
MARKERS = matplotlib.markers.MarkerStyle.markers.keys()[5:]
import numpy as np

sys.path.append('./utils')
from py_utils import get_spaced_colors_util

if __name__ == '__main__':

    results = pd.read_pickle('library/results_data_frame_10fish.pkl')
    preprocessings = ['body', 'body_blob', 'portrait']
    plt.ion()
    fig, ax_arr = plt.subplots(1,3,sharey=True)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x/100,screen_y/100))


    for j, preprocessing in enumerate(preprocessings):
        print(j)
        tests_names = results[results.preprocessing_type == preprocessing].test_name.unique()[::-1]
        print(tests_names)
        frames_per_fragment_conditions = list(results.frames_per_fragment.unique())
        num_frames_conditions = len(frames_per_fragment_conditions)
        num_repetitions = len(list(results.repetition.unique()))
        RGB_tuples = get_spaced_colors_util(num_frames_conditions, norm=True, black=False)
        epsilon = [-0.01 , -0.005, 0., 0.005, 0.01]
        epsilon = [-0.1 , -0.05, 0., 0.05, 0.1]
        epsilon = [0. , 0., 0., 0., 0.]

        acc = np.ones((4,num_frames_conditions,num_repetitions))*np.nan
        for i, test_name in enumerate(tests_names):
            results_test = results[results.test_name == test_name]
            acc[i,:,:] = np.reshape(np.asarray(results_test.accuracy),(num_frames_conditions,num_repetitions))

        if preprocessing == 'body':
            new_results = pd.read_pickle('library/results_data_frame_10fish_body_noFrozenPretrain.pkl')
            test_name = new_results.test_name.unique()[0]
            results_test = new_results[new_results.test_name == test_name]
            acc[3,:,:] = np.reshape(np.asarray(results_test.accuracy),(num_frames_conditions,num_repetitions))

        for i, mean_frames in enumerate(frames_per_fragment_conditions):
            accuracies = np.squeeze(acc[:,i,:])
            acc_median = np.nanmedian(accuracies,axis = 1)
            ax_arr[j].plot(np.asarray([0,1,2,3]),acc_median,label = str(mean_frames), color = np.asarray(RGB_tuples[i]), linewidth = 2, marker = MARKERS[i])
            for k in range(10):
                ax_arr[j].scatter(np.asarray([0,1,2,3])+epsilon[i], accuracies[:,k], color = np.asarray(RGB_tuples[i]), alpha = .3, marker = MARKERS[i])


        ax_arr[j].set_xlabel('Algorithm protocol')
        if j == 2:
            ax_arr[j].legend(title="mean number of \nframes in \nindividual fragments", fancybox=True)
        if j == 0:
            ax_arr[j].set_ylabel('accuracy')

        ax_arr[j].set_ylim(0.75,1.01)
        ax_arr[j].set_xticks([0,1,2,3])
        ax_arr[j].set_xticklabels([ 'noPretain\nnoAccum',
                                    'noPretrain\nAccum',
                                    'Pretrain\nAccum',
                                    'NoFrozenPretrain\nAccum'])
        ax_arr[j].set_title(preprocessing)
    fig.savefig('10fish_library_tests_algorithm_protocols_differentPreprocessings.png', transparent=True)
