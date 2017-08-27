from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys
sys.setrecursionlimit(100000)
import glob
import numpy as np
import cPickle as pickle

# Import third party libraries
import cv2
from pprint import pprint
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns


# Import application/library specifics
sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./')
# sys.path.append('IdTrackerDeep/tracker')

from video import Video
from blob import compute_fragment_identifier_and_blob_index,\
                connect_blob_list,\
                apply_model_area_to_video,\
                ListOfBlobs,\
                get_images_from_blobs_in_video,\
                reset_blobs_fragmentation_parameters
from globalfragment import  give_me_list_of_global_fragments,\
                            ModelArea,\
                            give_me_pre_training_global_fragments,\
                            get_images_and_labels_from_global_fragments,\
                            subsample_images_for_last_training,\
                            order_global_fragments_by_distance_travelled,\
                            compute_and_plot_global_fragments_statistics
from GUI_utils import selectDir

if __name__ == '__main__':

    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)
    #change this
    # blobs_path = '/media/atlas/idTrackerDeep_LargeGroups_3/idTrackerDeep_LargeGroups/TU20170307/numberIndivs_100/First/session_1/preprocessing/blobs_collection.npy'
    # global_fragments_path = '/media/atlas/idTrackerDeep_LargeGroups_3/idTrackerDeep_LargeGroups/TU20170307/numberIndivs_100/First/session_1/preprocessing/global_fragments.npy'
    blobs_path = video.blobs_path
    global_fragments_path = video.global_fragments_path
    list_of_blobs = ListOfBlobs.load(blobs_path)
    blobs = list_of_blobs.blobs_in_video
    print("loading global fragments")
    global_fragments = np.load(global_fragments_path)

    compute_and_plot_global_fragments_statistics(video, blobs, global_fragments)
