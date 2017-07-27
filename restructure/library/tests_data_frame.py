from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys

import pandas as pd

def tests_data_frame():
    tests_data_frame = pd.DataFrame()

    """
    'test_name': (string) name of the test
    'knowledge_transfer_flag': (bool) perform knowledge transfer from other model or not
    'knowledge_transfer_folder': (string) path to the model where to perform knowledge transfer from
    'pretraining_flag': (bool) perform pretraining or not
    'percentage_of_fragments_in_pretraining': (float [0. 1.]) percentage of global fragments used for pretraining
    'train_filters_in_accumulation': (bool) train filter during the accumulation process
    'accumulation_certainty': (float [0. 1.]) threshold certainty
    'IMDB_codes': (string) letters of the libraries used for the test
    'ids_codes': (string) f = first part of the library, s = second part of the library, a = all the library. len(ids_codes) should be len(IMDB_codes)
    'group_sizes': (list) group sizes for the test
    'frames_in_video': (list) number of frames per video to be tested
    'frames_per_individual_fragment': (list) lenght of frames in individual fragments to be tested
    'repetitions': (list) repetitions to be run (note that the repetition number is the seed of the random generator for the different random processes in the test)
    """

    ''' ************************************************************************
    Uncorrelated images test with different networks
    *************************************************************************'''
    preprocessing_types = ['portrait', 'body', 'body_blob']
    IMDB_codes = ['ABC','DEF','GHI']
    CNN_models = [0,1,2,3,4,5]
    for preprocessing, IMBD_code in zip(preprocessing_types,IMDB_codes):
        for CNN_model in CNN_models:
            tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_DEF_aaa_cnn_' + str(CNN_model) + '_' + preprocessing,
                                                            "CNN_model": CNN_model,
                                                            "knowledge_transfer_flag": False,
                                                            "knowledge_transfer_folder": '',
                                                            "pretraining_flag": False,
                                                            "percentage_of_frames_in_pretaining": 0.,
                                                            "only_accumulate_one_fragment": False,
                                                            "train_filters_in_accumulation": False,
                                                            "accumulation_certainty": 0.,
                                                            "preprocessing_type": preprocessing,
                                                            "IMDB_codes": IMBD_code,
                                                            "ids_codes": 'aaa',
                                                            "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                            "frames_in_video": [3000],
                                                            "frames_per_individual_fragment": [0],
                                                            "repetitions": [1, 2, 3],
                                                             }, ignore_index=True)

    ''' ************************************************************************
    Tests with correlated images and the real algorithm
    Portrait preprocessing
    100 fish 3000 frames per video. Normal distribution of individual fragments
    *************************************************************************'''
    solve_duplications = [True, False]
    preprocessing_types = ['portrait', 'body', 'body_blob']
    IMDB_codes = ['ABC','DEF','GHI']
    test_names = ['correlated_images_DEF_aaa_CNN0_noPretrain_noAccum_10fish_3000frames_gamma',
                    'correlated_images_DEF_aaa_CNN0_noPretrain_Accum05_10fish_3000frames_gamma',
                    'correlated_images_DEF_aaa_CNN0_Pretrain_Accum05_10fish_3000frames_gamma'
                    ]
    pretraining_flags = [False, False, True]
    percentage_of_frames_in_pretainings = [0,0,0.25]
    only_accumulate_one_fragments = [True, False, False]
    train_filters_in_accumulations = [True, True, False]
    accumulation_certainties = [.1, .5, .5]

    for solve_duplication in solve_duplications:
        if solve_duplication:
            duplications_extension = 'noDuplications'
        else:
            duplications_extension = 'duplications'
        for preprocessing, IMBD_code in zip(preprocessing_types,IMDB_codes):
            for i, test_name in enumerate(test_names):
                tests_data_frame = tests_data_frame.append({"test_name": test_name + '_' + preprocessing + '_' + duplications_extension,
                                                                "CNN_model": 0,
                                                                "knowledge_transfer_flag": False,
                                                                "knowledge_transfer_folder": '',
                                                                "pretraining_flag": pretraining_flags[i],
                                                                "percentage_of_frames_in_pretaining": percentage_of_frames_in_pretainings[i],
                                                                "only_accumulate_one_fragment": only_accumulate_one_fragments[i],
                                                                "train_filters_in_accumulation": train_filters_in_accumulations[i],
                                                                "accumulation_certainty": accumulation_certainties[i],
                                                                "solve_duplications": solve_duplication,
                                                                "preprocessing_type": preprocessing,
                                                                "IMDB_codes": IMBD_code,
                                                                "ids_codes": 'aaa',
                                                                "group_sizes": [100],
                                                                "frames_in_video": [3000],
                                                                "frames_per_individual_fragment": [50, 165, 848.5],
                                                                "std_frames_per_individual_fragment": [1.5, 1, .6], # it is the parameter of the gamma related to the variance "a" in scipy
                                                                "repetitions": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                                 }, ignore_index=True)

    tests_data_frame.to_pickle('./library/tests_data_frame.pkl')
    return tests_data_frame

if __name__ == '__main__':
    tests_data_frame = tests_data_frame()
