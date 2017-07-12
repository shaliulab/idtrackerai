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
    Portrait preprocessing (libraries DEF)
    *************************************************************************'''

    # 0 uncorrelated_images DEF_aaa cnn_model_0
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_DEF_aaa_cnn_0',
                                                    "CNN_model": 0,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 1 uncorrelated_images DEF_aaa cnn_model_1
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_DEF_aaa_cnn_1',
                                                    "CNN_model": 1,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 2 uncorrelated_images DEF_aaa cnn_model_2
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_DEF_aaa_cnn_2',
                                                    "CNN_model": 2,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 3 uncorrelated_images DEF_aaa cnn_model_3
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_DEF_aaa_cnn_3',
                                                    "CNN_model": 3,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 4 uncorrelated_images DEF_aaa cnn_model_4
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_DEF_aaa_cnn_4',
                                                    "CNN_model": 4,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 5 uncorrelated_images DEF_aaa cnn_model_5
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_DEF_aaa_cnn_5',
                                                    "CNN_model": 5,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    ''' ************************************************************************
    Uncorrelated images test with different networks
    Portrait preprocessing (libraries GHI)
    *************************************************************************'''

    # 6 uncorrelated_images GHI_aaa cnn_model_0
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_GHI_aaa_cnn_0',
                                                    "CNN_model": 0,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "animal_type": 'fly',
                                                    "IMDB_codes": 'GHI',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 7 uncorrelated_images GHI_aaa cnn_model_1
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_GHI_aaa_cnn_1',
                                                    "CNN_model": 1,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "animal_type": 'fly',
                                                    "IMDB_codes": 'GHI',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 8 uncorrelated_images GHI_aaa cnn_model_2
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_GHI_aaa_cnn_2',
                                                    "CNN_model": 2,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "animal_type": 'fly',
                                                    "IMDB_codes": 'GHI',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 9 uncorrelated_images GHI_aaa cnn_model_3
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_GHI_aaa_cnn_3',
                                                    "CNN_model": 3,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "animal_type": 'fly',
                                                    "IMDB_codes": 'GHI',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 10 uncorrelated_images GHI_aaa cnn_model_4
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_GHI_aaa_cnn_4',
                                                    "CNN_model": 4,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "animal_type": 'fly',
                                                    "IMDB_codes": 'GHI',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    # 11 uncorrelated_images GHI_aaa cnn_model_5
    tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_GHI_aaa_cnn_5',
                                                    "CNN_model": 5,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0.,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.,
                                                    "animal_type": 'fly',
                                                    "IMDB_codes": 'GHI',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                    "frames_in_video": [3000],
                                                    "frames_per_individual_fragment": [0],
                                                    "repetitions": [1, 2, 3],
                                                     }, ignore_index=True)

    ''' ************************************************************************
    Tests with correlated images and the real algorithm
    Portrait preprocessing (libaries DEF)
    1000 fish 3000 frames per vidio. Normal distribution of individual fragments
    *************************************************************************'''
    group_sizes = [100]
    frames_in_video = [3000]
    frames_per_individual_fragment = [50, 100, 250, 500, 1000]
    repetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 12 correlated_images noPretrain noAccum Gaussian distribution of individual fragments
    tests_data_frame = tests_data_frame.append({"test_name": 'correlated_images_DEF_aaa_CNN0_noPretrain_noAccum_100fish_3000frames_gaussian',
                                                    "CNN_model": 0,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0,
                                                    "only_accumulate_one_fragment": True,
                                                    "train_filters_in_accumulation": True,
                                                    "accumulation_certainty": 0.1,
                                                    "animal_type": 'fish',
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": group_sizes,
                                                    "frames_in_video": frames_in_video,
                                                    "frames_per_individual_fragment": frames_per_individual_fragment,
                                                    "std_frames_per_individual_fragment": 150,
                                                    "repetitions": repetitions,
                                                     }, ignore_index=True)

    # 13 correlated_images noPretrain Accum certainty 0.1 Gaussian distribution of individual fragments
    tests_data_frame = tests_data_frame.append({"test_name": 'correlated_images_DEF_aaa_CNN0_noPretrain_Accum01_100fish_3000frames_gaussian',
                                                    "CNN_model": 0,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": True,
                                                    "accumulation_certainty": 0.1,
                                                    "animal_type": 'fish',
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": group_sizes,
                                                    "frames_in_video": frames_in_video,
                                                    "frames_per_individual_fragment": frames_per_individual_fragment,
                                                    "std_frames_per_individual_fragment": 150,
                                                    "repetitions": repetitions,
                                                     }, ignore_index=True)


    # 14 correlated_images noPretrain Accum certainty 0.5 Gaussian distribution of individual fragments
    tests_data_frame = tests_data_frame.append({"test_name": 'correlated_images_DEF_aaa_CNN0_noPretrain_Accum05_100fish_3000frames_gaussian',
                                                    "CNN_model": 0,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": True,
                                                    "accumulation_certainty": 0.5,
                                                    "animal_type": 'fish',
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": group_sizes,
                                                    "frames_in_video": frames_in_video,
                                                    "frames_per_individual_fragment": frames_per_individual_fragment,
                                                    "std_frames_per_individual_fragment": 150,
                                                    "repetitions": repetitions,
                                                     }, ignore_index=True)

    # 15 correlated_images noPretrain Accum certainty 0.9 Gaussian distribution of individual fragments
    tests_data_frame = tests_data_frame.append({"test_name": 'correlated_images_DEF_aaa_CNN0_noPretrain_Accum09_100fish_3000frames_gaussian',
                                                    "CNN_model": 0,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": False,
                                                    "percentage_of_frames_in_pretaining": 0,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": True,
                                                    "accumulation_certainty": 0.9,
                                                    "animal_type": 'fish',
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": group_sizes,
                                                    "frames_in_video": frames_in_video,
                                                    "frames_per_individual_fragment": frames_per_individual_fragment,
                                                    "std_frames_per_individual_fragment": 150,
                                                    "repetitions": repetitions,
                                                     }, ignore_index=True)

    # 16 correlated_images Pretrain noAccum Gaussian distribution of individual fragments
    tests_data_frame = tests_data_frame.append({"test_name": 'correlated_images_DEF_aaa_CNN0_Pretrain_noAccum_100fish_3000frames_gaussian',
                                                    "CNN_model": 0,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": True,
                                                    "percentage_of_frames_in_pretaining": 0.25,
                                                    "only_accumulate_one_fragment": True,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.5,
                                                    "animal_type": 'fish',
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": group_sizes,
                                                    "frames_in_video": frames_in_video,
                                                    "frames_per_individual_fragment": frames_per_individual_fragment,
                                                    "std_frames_per_individual_fragment": 150,
                                                    "repetitions": repetitions,
                                                     }, ignore_index=True)

    # 17 correlated_images Pretrain Accum certainty 0.5 Gaussian distribution of individual fragments
    tests_data_frame = tests_data_frame.append({"test_name": 'correlated_images_DEF_aaa_CNN0_Pretrain_Accum05_100fish_3000frames_gaussian',
                                                    "CNN_model": 0,
                                                    "knowledge_transfer_flag": False,
                                                    "knowledge_transfer_folder": '',
                                                    "pretraining_flag": True,
                                                    "percentage_of_frames_in_pretaining": 0.25,
                                                    "only_accumulate_one_fragment": False,
                                                    "train_filters_in_accumulation": False,
                                                    "accumulation_certainty": 0.5,
                                                    "animal_type": 'fish',
                                                    "IMDB_codes": 'DEF',
                                                    "ids_codes": 'aaa',
                                                    "group_sizes": group_sizes,
                                                    "frames_in_video": frames_in_video,
                                                    "frames_per_individual_fragment": frames_per_individual_fragment,
                                                    "std_frames_per_individual_fragment": 150,
                                                    "repetitions": repetitions,
                                                     }, ignore_index=True)

    tests_data_frame.to_pickle('./library/tests_data_frame.pkl')
    return tests_data_frame

if __name__ == '__main__':
    tests_data_frame = tests_data_frame()
