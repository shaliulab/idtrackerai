from __future__ import absolute_import, division, print_function
import os
import sys

import numpy as np
from pprint import pprint
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    path_to_results_hard_drive = '/media/rhea/ground_truth_results_backup'
    path_to_tracked_videos_data_frame = os.path.join(path_to_results_hard_drive,'tracked_videos/tracked_videos_data_frame.pkl')
    if os.path.isfile(path_to_tracked_videos_data_frame):
        tracked_videos_data_frame = pd.read_pickle(path_to_tracked_videos_data_frame)
        tracked_videos_data_frame_good = tracked_videos_data_frame[tracked_videos_data_frame.bad_video_example == False]
        sns.set_style("ticks")
        plt.ion()
        fig, ax = plt.subplots(1,1)
        window = plt.get_current_fig_manager().window
        screen_y = window.winfo_screenheight()
        screen_x = window.winfo_screenwidth()
        fig.set_size_inches((screen_x/100/3,screen_y/100/2))
        fig.subplots_adjust(left=None, bottom=.15, right=None, top=None,
                    wspace=None, hspace=None)
        ax.plot(tracked_videos_data_frame.estimated_accuracy_in_validated_part*100,
                tracked_videos_data_frame.accuracy_identification_and_interpolation*100,
                'o', alpha = .7,
                markersize = 10)
        ax.plot([0,100],[0,100],'r--')
        plt.axis('square')
        ax.set_xlim((99.0,100.01))
        ax.set_ylim((99.0,100.01))
        ax.set_yticks(np.linspace(99.0,100,5))
        ax.set_yticklabels(np.linspace(99.0,100,5))
        ax.set_xticks(np.linspace(99.0,100,5))
        ax.set_xticklabels(np.linspace(99.0,100,5))
        ax.set_xlabel('Estimated accuracy', fontsize = 20)
        ax.set_ylabel('Groundtruth accuracy', fontsize = 20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        sns.despine(ax = ax, right = True, top = True)
        plt.show()
        fig.savefig(os.path.join(path_to_results_hard_drive,'tracked_videos/estimated_accuracy_vs_accuracy.pdf'), transparent=True)
    else:
        print("%s not found. Please build it using the script build_summary_videos_data_frame.py" %path_to_tracked_videos_data_frame)
