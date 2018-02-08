from __future__ import absolute_import, division, print_function
import os
import sys

import numpy as np
from pprint import pprint
import pandas as pd

sessions_smaller_groups = ['10_fish_group4/first/session_20180122',
                 '10_fish_group5/first/session_20180131',
                 '10_fish_group6/first/session_20180202',
                 '10_flies_compressed_clara/session_20180207',
                 'idTrackerVideos/Hipertec_pesados/Medaka/2012may31/Grupo10/session_20180201',
                 'idTrackerVideos/Hipertec_pesados/Medaka/2012may31/Grupo5/session_20180131',
                 'ants_andrew_1/session_20180206',
                 'idTrackerVideos/Moscas/2011dic12/Video_4fem_2mal_bottom/session_20180130',
                 'idTrackerVideos/Moscas/20121010/PlatoGrande_8females_2/session_20180131',
                 'idTrackerVideos/NatureMethods/Isogenicos/Wik_8_grupo4/session_20180130',
                 'idTrackerVideos/NatureMethods/Ratones4/session_20180205',
                 'idTrackerVideos/NatureMethods/VideoRatonesDespeinaos3/session_20180206',
                 'idTrackerVideos/Ratones/20121203/2aguties/session_20180204',
                 'idTrackerVideos/Ratones/20121203/2negroscanosos/session_20180204',
                 'idTrackerVideos/Ratones/20121203/2negroslisocanoso/session_20180205',
                 'idTrackerVideos/Ratones/20121203/2negroslisos/session_20180205',
                 'idTrackerVideos/ValidacionTracking/Moscas/Platogrande_8females/session_20180131',
                 'idTrackerVideos/Zebrafish_nacreLucie/pair3ht/session_20180207']

if __name__ == '__main__':

    path_to_tracked_videos_data_frame = '/media/chronos/ground_truth_results_backup/tracked_videos_data_frame.pkl'

    tracked_videos_data_frame = pd.read_pickle(path_to_tracked_videos_data_frame)
