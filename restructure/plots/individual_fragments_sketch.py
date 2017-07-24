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
from matplotlib.patches import Rectangle
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
                            order_global_fragments_by_distance_travelled
from segmentation import segment
from GUI_utils import selectFile,\
                    getInput,\
                    selectOptions,\
                    ROISelectorPreview,\
                    selectPreprocParams,\
                    fragmentation_inspector,\
                    frame_by_frame_identity_inspector,\
                    selectDir
from py_utils import getExistentFiles, get_spaced_colors_util
from video_utils import checkBkg
from pre_trainer import pre_train
from accumulation_manager import AccumulationManager
from network_params import NetworkParams
from trainer import train
from assigner import assign,\
                    assign_identity_to_blobs_in_video,\
                    compute_P1_for_blobs_in_video,\
                    assign_identity_to_blobs_in_video_by_fragment
from visualize_embeddings import visualize_embeddings_global_fragments
from id_CNN import ConvNetwork

if __name__ == '__main__':

    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)
    #change this
    blobs_path = video.blobs_path
    global_fragments_path = video.global_fragments_path
    list_of_blobs = ListOfBlobs.load(blobs_path)
    blobs = list_of_blobs.blobs_in_video
    print("loading global fragments")
    global_fragments = np.load(global_fragments_path)

    global_fragment = global_fragments[10]
    global_fragment.compute_start_end_frame_indices_of_individual_fragments(blobs)
    strats_ends_individual_fragments = np.asarray(global_fragment.starts_ends_individual_fragments)
    print(strats_ends_individual_fragments)
    min_start_individual_fragments = np.min(strats_ends_individual_fragments[:,0])
    max_end_individual_fragments = np.max(strats_ends_individual_fragments[:,1])
    max_start_individual_fragments = np.max(strats_ends_individual_fragments[:,0])
    min_end_individual_fragments = np.min(strats_ends_individual_fragments[:,1])
    blob_indices_individial_fragments = np.asarray([(blob._blob_index,fragment_identifier) for blob, fragment_identifier in zip(blobs[global_fragment.index_beginning_of_fragment],global_fragment.individual_fragments_identifiers)])


    ''' global fragment sketch'''
    plt.ion()
    fig, ax_arr = plt.subplots(4,1)
    colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=True)

    ax_arr[0].add_patch(Rectangle((max_start_individual_fragments-.5, 0.5),min_end_individual_fragments-max_start_individual_fragments + 1,video.number_of_animals ,alpha=1, fc = '0.85'))
    ax_arr[1].add_patch(Rectangle((max_start_individual_fragments-.5, 0.5),min_end_individual_fragments-max_start_individual_fragments + 1,video.number_of_animals ,alpha=1, fc = '0.85'))
    ax_arr[2].add_patch(Rectangle((max_start_individual_fragments-.5, 0.5),min_end_individual_fragments-max_start_individual_fragments + 1,video.number_of_animals ,alpha=1, fc = '0.85'))
    ax_arr[3].add_patch(Rectangle((max_start_individual_fragments-.5, 0.5),min_end_individual_fragments-max_start_individual_fragments + 1,video.number_of_animals ,alpha=1, fc = '0.85'))

    for i in range(min_start_individual_fragments, max_end_individual_fragments):
        blobs_in_frame = blobs[i]
        for j, blob in enumerate(blobs_in_frame):
            next_blobs = blob.next
            for next_blob in next_blobs:

                # individual fragments and crossings
                blob_index_next = blobs[i+1].index(next_blob) + 1
                if blob._identity == 0:
                    ax_arr[0].plot([i, i+1],[j + 1,blob_index_next], 'o-' ,c = 'k', markersize = 3)
                else:
                    ax_arr[0].plot([i, i+1],[j + 1,blob_index_next], 'o-' ,c = '.75', markersize = 3)

                # hightlight invidiaul fragments in globl fragment
                if blob._fragment_identifier in global_fragment.individual_fragments_identifiers:
                    ax_arr[1].plot([i, i+1],[j + 1, blob_index_next], 'o-' ,c = colors[blob._identity], markersize = 3)
                else:
                    if blob._identity == 0:
                        ax_arr[1].plot([i, i+1],[j + 1,blob_index_next], 'o-' ,c = 'k', markersize = 3)
                    else:
                        ax_arr[1].plot([i, i+1],[j + 1,blob_index_next], 'o-' ,c = '.75', markersize = 3)

                # Extact individual fragments from global fragmtn
                blob_index_next = blobs[i+1].index(next_blob) + 1
                if blob._fragment_identifier in global_fragment.individual_fragments_identifiers and next_blob.is_a_fish:
                    ax_arr[2].plot([i, i+1],[j + 1, blob_index_next], 'o-' ,c = colors[blob._identity], markersize = 3)

                # Unroll hierarchies
                if blob._fragment_identifier in global_fragment.individual_fragments_identifiers and next_blob.is_a_fish:
                    blob_index = blob_indices_individial_fragments[np.where(blob_indices_individial_fragments[:,1] == blob._fragment_identifier)[0],:][0][0] + 1
                    ax_arr[3].plot([i,i+1],[blob_index, blob_index], '-' ,c = colors[blob._identity], linewidth = 10)
                    # ax_arr[3].plot([i,i+1],[blob_index, blob_index], 'o-' ,c = colors[blob._identity], markersize = 3, solid_capstyle ='butt')

    ax_arr[0].set_yticks(range(1,video.number_of_animals+1),range(1,video.number_of_animals+1))
    ax_arr[0].set_yticks(list(range(1,video.number_of_animals+1)))
    ax_arr[0].set_yticklabels(list(range(1,video.number_of_animals+1)))
    ax_arr[1].set_ylabel('segmentation hierarchy index')
    ax_arr[1].set_yticks(list(range(1,video.number_of_animals+1)))
    ax_arr[1].set_yticklabels(list(range(1,video.number_of_animals+1)))
    ax_arr[2].set_yticks(list(range(1,video.number_of_animals+1)))
    ax_arr[2].set_yticklabels(list(range(1,video.number_of_animals+1)))
    ax_arr[3].set_ylabel('blob index')
    ax_arr[3].set_xlabel('frame number')
    ax_arr[3].set_yticks(list(range(1,video.number_of_animals+1)))
    ax_arr[3].set_yticklabels(list(range(1,video.number_of_animals+1)))

    ''' 3d trajectories '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    number_of_frames = max_end_individual_fragments - min_start_individual_fragments
    centroid_trajectories = np.ones((number_of_frames, video.number_of_animals, 2))*np.NaN

    for i, frame_number in enumerate(range(min_start_individual_fragments, max_end_individual_fragments)):
        blobs_in_frame = blobs[frame_number]
        for j, blob in enumerate(blobs_in_frame):
            if blob._fragment_identifier in global_fragment.individual_fragments_identifiers and next_blob.is_a_fish:
                centroid_trajectories[i, blob.identity-1, :] = blob.centroid

    for individual in range(video.number_of_animals):
        ax.plot(centroid_trajectories[:,individual,0], centroid_trajectories[:,individual,1], range(number_of_frames),color = colors[individual + 1] )
