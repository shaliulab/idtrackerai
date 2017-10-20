from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys

import glob
import numpy as np
import cPickle as pickle

# Import third party libraries
import cv2
from pprint import pprint
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import seaborn as sns


# Import application/library specifics
sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./')
# sys.path.append('IdTrackerDeep/tracker')

from globalfragment import  give_me_list_of_global_fragments,\
                            ModelArea,\
                            give_me_pre_training_global_fragments,\
                            get_images_and_labels_from_global_fragments,\
                            order_global_fragments_by_distance_travelled

if __name__ == '__main__':

    global_fragments_paths = ['/media/rhea/idTrackerDeep_LargeGroups_1/idTrackerDeep_LargeGroups/TU20170307/numberIndivs_100/First/session_1/preprocessing/global_fragments.npy',
                                '/media/rhea/idTrackerDeep_LargeGroups_1/idTrackerDeep_LargeGroups/TU20170307/numberIndivs_100/Second/session_1/preprocessing/global_fragments.npy' ]

    # individual fragments statistics
    individual_fragments_added = []
    number_of_frames_in_individual_fragments = []

    ''' plotting '''
    sns.set_style("ticks")
    fig, ax = plt.subplots(1)
    #window = plt.get_current_fig_manager().window
    #screen_y = window.winfo_screenheight()
    #screen_x = window.winfo_screenwidth()
    #fig.set_size_inches((screen_x/100,screen_y/100))
    labels = ['schooling', 'shoaling']
    colors = ['r', 'b']
    for j, global_fragments_path in enumerate(global_fragments_paths):
        print("loading global fragments")
        global_fragments = np.load(global_fragments_path)
        individual_fragments_added = []
        number_of_frames_in_individual_fragments = []
        for global_fragment in global_fragments:
            for i, individual_fragment_identifier in enumerate(global_fragment.individual_fragments_identifiers):
                if individual_fragment_identifier not in individual_fragments_added:
                    individual_fragments_added.append(individual_fragment_identifier)
                    number_of_frames_in_individual_fragments.append(global_fragment._number_of_images_per_individual_fragment[i])
            np.save(labels[j] + '_individual_fragments_distribution.npy',number_of_frames_in_individual_fragments)
        # remove global fragments that are lenght 0
        number_of_frames_in_individual_fragments = np.asarray(filter(lambda x: x >= 3, number_of_frames_in_individual_fragments))

        # number of frames in individual fragments
        nbins = 20
        MIN = np.min(number_of_frames_in_individual_fragments)
        MAX = np.max(number_of_frames_in_individual_fragments)
        hist, bin_edges = np.histogram(number_of_frames_in_individual_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
        ax.semilogx(bin_edges[:-1], hist ,markersize = 5, label = labels[j], color = colors[j], linewidth = 2)

    ax.legend(title="behavior type", fancybox=True)
    ax.set_xlabel('number of frames')
    ax.set_ylabel('number of individual fragments')

    fig.savefig('schooling_vs_shoaling_individual_fragments.pdf', transparent=True)
