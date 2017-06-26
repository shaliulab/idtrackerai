from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys

import pandas as pd

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

# 0 uncorrleaged_images DEF_aaa cnn_model_0
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

# 1 uncorrleaged_images DEF_aaa cnn_model_1
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

# 2 uncorrleaged_images DEF_aaa cnn_model_2
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

# 3 uncorrleaged_images DEF_aaa cnn_model_3
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

# 4 uncorrleaged_images DEF_aaa cnn_model_4
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

# 5 uncorrleaged_images DEF_aaa cnn_model_5
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

# # uncorrleaged_images DEF_aaa cnn_model_0
# tests_data_frame = tests_data_frame.append({"test_name": 'uncorrelated_images_DEF_aaa_CNN0',
#                                                 "CNN_model": 0,
#                                                 "knowledge_transfer_flag": False,
#                                                 "knowledge_transfer_folder": '',
#                                                 "pretraining_flag": False,
#                                                 "percentage_of_frames_in_pretaining": 0.,
#                                                 "only_accumulate_one_fragment": False,
#                                                 "train_filters_in_accumulation": False,
#                                                 "accumulation_certainty": 0.,
#                                                 "IMDB_codes": 'DEF',
#                                                 "ids_codes": 'aaa',
#                                                 "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
#                                                 "frames_in_video": [35000],
#                                                 "frames_per_individual_fragment": [0],
#                                                 "repetitions": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                                                  }, ignore_index=True)
#
# # only_one_global_fragment_for_training
# tests_data_frame = tests_data_frame.append({"test_name": 'Only_one_global_fragment_for_training',
#                                                 "CNN_model": 0,
#                                                 "knowledge_transfer_flag": False,
#                                                 "knowledge_transfer_folder": '',
#                                                 "pretraining_flag": False,
#                                                 "percentage_of_frames_in_pretaining": 1.,
#                                                 "only_accumulate_one_fragment": False,
#                                                 "train_filters_in_accumulation": False,
#                                                 "accumulation_certainty": .1,
#                                                 "IMDB_codes": 'DEF',
#                                                 "ids_codes": 'aaa',
#                                                 "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
#                                                 "frames_in_video": [5000],
#                                                 "frames_per_individual_fragment": [5, 10, 20, 40, 80, 160, 320, 640],
#                                                 "repetitions": [1, 2],
#                                                  }, ignore_index=True)
#
# # all_accumulation
# tests_data_frame = tests_data_frame.append({"test_name": 'all_accumulation',
#                                                 "CNN_model": 0,
#                                                 "knowledge_transfer_flag": False,
#                                                 "knowledge_transfer_folder": '',
#                                                 "pretraining_flag": False,
#                                                 "percentage_of_frames_in_pretaining": 1.,
#                                                 "only_accumulate_one_fragment": False,
#                                                 "train_filters_in_accumulation": False,
#                                                 "accumulation_certainty": .1,
#                                                 "IMDB_codes": 'DEF',
#                                                 "ids_codes": 'aaa',
#                                                 "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
#                                                 "frames_in_video": [5000],
#                                                 "frames_per_individual_fragment": [5, 10, 20, 40, 80, 160, 320, 640],
#                                                 "repetitions": [1, 2],
#                                                  }, ignore_index=True)
#
# # pretraining_and_accumulation
# tests_data_frame = tests_data_frame.append({"test_name": 'pretraining_and_accumulation',
#                                                 "CNN_model": 0,
#                                                 "knowledge_transfer_flag": False,
#                                                 "knowledge_transfer_folder": '',
#                                                 "pretraining_flag": True,
#                                                 "percentage_of_fragments_in_pretraining": 1.,
#                                                 "only_accumulate_one_fragment": False,
#                                                 "train_filters_in_accumulation": False,
#                                                 "accumulation_certainty": .1,
#                                                 "IMDB_codes": 'DEF',
#                                                 "ids_codes": 'aaa',
#                                                 "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
#                                                 "frames_in_video": [5000],
#                                                 "frames_per_individual_fragment": [5, 10, 20, 40, 80, 160, 320, 640],
#                                                 "repetitions": [1, 2],
#                                                  }, ignore_index=True)


tests_data_frame.to_pickle('./library/tests_data_frame.pkl')
