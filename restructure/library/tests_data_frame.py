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

# test 1
tests_data_frame = tests_data_frame.append({"test_name": 'testing',
                                                "only_accumulate_one_fragment": True,
                                                "knowledge_transfer_flag": False,
                                                "knowledge_transfer_folder": '',
                                                "pretraining_flag": False,
                                                "percentage_of_fragments_in_pretraining": 1.,
                                                "train_filters_in_accumulation": False,
                                                "accumulation_certainty": .1,
                                                "IMDB_codes": 'D',
                                                "ids_codes": 'aa',
                                                "group_sizes": [2, 5, 10, 30, 60, 80, 100, 150],
                                                "frames_in_video": [5000],
                                                "frames_per_individual_fragment": [5, 10, 20, 40, 80, 160],
                                                "repetitions": [1, 2],
                                                 }, ignore_index=True)


tests_data_frame.to_pickle('./library/tests_data_frame.pkl')
