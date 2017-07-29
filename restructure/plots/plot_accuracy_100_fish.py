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

    results = pd.read_pickle('library/results_data_frame_gamma_100fish_posterBehav2017_2.pkl')
    tests_names = [ 'correlated_images_DEF_aaa_CNN0_noPretrain_noAccum_10fish_3000frames_gamma_portrait_duplications',
                   'correlated_images_DEF_aaa_CNN0_noPretrain_Accum05_10fish_3000frames_gamma_portrait_duplications',
                   'correlated_images_DEF_aaa_CNN0_Pretrain_Accum05_10fish_3000frames_gamma_portrait_duplications']
    frames_per_fragment_conditions = [50.0, 250.0, 165.0, 848.5][::-1]
    num_frames_conditions = len(frames_per_fragment_conditions)
    num_repetitions = len(list(results.repetition.unique()))

    acc = np.ones((3,4,5))
    for i, test_name in enumerate(tests_names):
        for j, frames_per_fragment in enumerate(frames_per_fragment_conditions):
            results_test = results[results.test_name == test_name]
            results_test = results_test[results_test.frames_per_fragment == frames_per_fragment]
            try:
                acc[i,j,:] = np.asarray(results_test.accuracy)
            except:
                print(test_name, frames_per_fragment)

    plt.ion()
    fig, ax = plt.subplots(1)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x*2/3/100,screen_y/100))

    # RGB_tuples = get_spaced_colors_util(num_frames_conditions, norm=True, black=False)
    RGB_tuples = ['y', 'g', 'r', 'b'][::-1]
    for i, mean_frames in enumerate(frames_per_fragment_conditions):
        accuracies = np.squeeze(acc[:,i,:])
        print(accuracies)
        acc_median = np.nanmean(accuracies,axis = 1)
        ax.plot(np.asarray([0,1,2]),acc_median,label = str(mean_frames), color = RGB_tuples[i], linewidth = 2, marker = MARKERS[i])
        for j in range(5):
            ax.plot(np.asarray([0,1,2]), accuracies[:,j], color = RGB_tuples[i], alpha = .3, linewidth = 1, marker = MARKERS[i])

    ax.legend(title="mean number of \nframes in \nindividual fragments", fancybox=True)
    ax.set_xlabel('Algorithm protocol')
    ax.set_ylabel('accuracy')
    plt.xticks([0,1,2], [ 'noPretain\nnoAccum',
                                'noPretrain\nAccum',
                                'Pretrain\nAccum'])
    ax.set_ylim(0.75,1.01)
    fig.savefig('100fish_library_tests_algorithm_protocols_portraits_gamma.pdf', transparent=True)
