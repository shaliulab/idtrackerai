from __future__ import absolute_import, division, print_function
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import collections
import pandas as pd
from pprint import pprint

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
                proportion_of_unique_fragments_for_accumulation = []
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
    test_name = 'correlated_images_DEF_aaa_CNN0_trainonly1GF'
    print('test to plot: ', test_name)
    test_dictionary = tests_data_frame.query('test_name == @test_name').to_dict()
    pprint(test_dictionary)

    # get results from the test to plot
    results_data_frame_test = results_data_frame.query('test_name == @test_name')
    results_data_frame_test['proportion_of_unique_fragments_for_accumulation'] = results_data_frame_test.number_of_unique_fragments / results_data_frame_test.number_of_candidate_fragments

    # average the repetitions
    repetition_averaged_data_frame = get_repetition_averaged_data_frame(results_data_frame_test)

    ########### plot
    plt.ion()
    # figure for accuracy and total time
    sns.set_style("ticks")
    fig1, ax_arr1 = plt.subplots(3,1)
    fig1.suptitle('%s - library %s - %i repetitions' %(test_name,
                                                    test_dictionary['IMDB_codes'].values()[0],
                                                    len(test_dictionary['repetitions'].values()[0])))


    # figure for repetitions and other statistics
    # fig2, ax_arr2 = plt.subplots(2,2)


    for frames_per_fragment in repetition_averaged_data_frame['frames_per_fragment'].unique():

        this_frames_per_fragment_data_frame = repetition_averaged_data_frame.query('frames_per_fragment == @frames_per_fragment')
        this_frames_per_fragment_data_frame = this_frames_per_fragment_data_frame.apply(pd.to_numeric, errors='ignore')

        accuracy = this_frames_per_fragment_data_frame.accuracy
        total_time = this_frames_per_fragment_data_frame.total_time
        proportion_of_unique_fragments_for_accumulation = this_frames_per_fragment_data_frame.proportion_of_unique_fragments_for_accumulation
        group_sizes = this_frames_per_fragment_data_frame.group_size.astype('float32')

        # accuracy
        ax = ax_arr1[0]
        ax.plot(group_sizes, accuracy, label = str(int(frames_per_fragment)))

        # total time
        ax = ax_arr1[1]
        ax.plot(group_sizes, total_time, label = str(int(frames_per_fragment)))

        # proportion_of_unique_fragments_for_accumulation
        ax = ax_arr1[2]
        ax.plot(group_sizes, proportion_of_unique_fragments_for_accumulation, label = str(int(frames_per_fragment)))



    ax = ax_arr1[0]
    ax.set_ylabel('accuracy')
    ax.set_ylim([0.,1.1])
    ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax.set_xticks(group_sizes)

    ax = ax_arr1[1]
    ax.set_ylabel('total time (sec)')
    ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax.set_xticks(group_sizes)

    ax = ax_arr1[2]
    ax.set_ylabel(r'$\%$' + ' of global fragments \nwithout repetitions')
    h_legend = ax.legend()
    h_legend.set_title('frames per \nindividual fragment')
    ax.set_xlim([0.,np.max(repetition_averaged_data_frame['group_size'])+2])
    ax.set_xticks(group_sizes)
    ax.set_xlabel('group size')


    plt.show()
