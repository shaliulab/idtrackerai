import os
import sys

import pandas as pd
from tests_data_frame import tests_data_frame

if __name__ == '__main__':
    list_of_data_frames_names_to_join = ['PCA', 'portraits']

    list_of_results_data_frames = []
    for data_frame_name in list_of_data_frames_names_to_join:
        list_of_results_data_frames.append(pd.read_pickle(os.path.join('./library','results_data_frame_' + data_frame_name + '.pkl')))


    results_data_frame = pd.concat(list_of_results_data_frames,ignore_index = True)
    results_data_frame.to_pickle('./library/results_data_frame.pkl')
    tests_data_frame = tests_data_frame()
