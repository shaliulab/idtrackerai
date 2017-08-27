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

    results = pd.read_pickle('library/results_data_frame_gamma_100fish_posterBehav2017.pkl')
    tests_names = [ 'correlated_images_DEF_aaa_CNN0_noPretrain_noAccum_10fish_3000frames_gamma_portrait_duplications',
                   'correlated_images_DEF_aaa_CNN0_noPretrain_Accum05_10fish_3000frames_gamma_portrait_duplications',
                   'correlated_images_DEF_aaa_CNN0_Pretrain_Accum05_10fish_3000frames_gamma_portrait_duplications']
    frames_per_fragment_conditions = list(results.frames_per_fragment.unique())
    num_frames_conditions = len(frames_per_fragment_conditions)
    num_repetitions = len(list(results.repetition.unique()))

    acc = np.ones((3,3,5))
    for i, test_name in enumerate(tests_names):
        results_test = results[results.test_name == test_name]
        if test_name == 'correlated_images_DEF_aaa_CNN0_Pretrain_Accum05_10fish_3000frames_gamma_portrait_duplications':
            acc[i,:2,:] = np.reshape(np.asarray(results_test.accuracy),(2,num_repetitions))
        else:
            acc[i,:,:] = np.reshape(np.asarray(results_test.accuracy),(num_frames_conditions,num_repetitions))

    plt.ion()
    fig, ax = plt.subplots(1)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x/100,screen_y/100))

    # RGB_tuples = get_spaced_colors_util(num_frames_conditions, norm=True, black=False)
    RGB_tuples = ['g', 'r', 'b']
    epsilon = [-0.01 , -0.005, 0., 0.005, 0.01]
    epsilon = [-0.1 , -0.05, 0., 0.05, 0.1]
    epsilon = [0. , 0., 0., 0., 0.]
    for i, mean_frames in enumerate(frames_per_fragment_conditions):
        accuracies = np.squeeze(acc[:,i,:])
        print(accuracies)
        acc_median = np.nanmedian(accuracies,axis = 1)
        ax.plot(np.asarray([0,1,2]),acc_median,label = str(mean_frames), color = RGB_tuples[i], linewidth = 2, marker = MARKERS[i])
        for j in range(5):
            ax.scatter(np.asarray([0,1,2])+epsilon[i], accuracies[:,j], color = RGB_tuples[i], alpha = .3, marker = MARKERS[i])

    ax.legend(title="mean number of \nframes in \nindividual fragments", fancybox=True)
    ax.set_xlabel('Algorithm protocol')
    ax.set_ylabel('accuracy')
    plt.xticks([0,1,2], [ 'noPretain\nnoAccum',
                                'noPretrain\nAccum',
                                'Pretrain\nAccum'])
    ax.set_ylim(0.75,1.01)
    fig.savefig('100fish_library_tests_algorithm_protocols_portraits_gamma.pdf', transparent=True)
