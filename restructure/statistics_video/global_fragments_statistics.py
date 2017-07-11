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
from py_utils import getExistentFiles
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
    blobs_path = '/media/atlas/idTrackerDeep_LargeGroups_3/idTrackerDeep_LargeGroups/TU20170307/numberIndivs_100/First/session_1/preprocessing/blobs_collection.npy'
    global_fragments_path = '/media/atlas/idTrackerDeep_LargeGroups_3/idTrackerDeep_LargeGroups/TU20170307/numberIndivs_100/First/session_1/preprocessing/global_fragments.npy'
    # blobs_path = video.blobs_path
    # global_fragments_path = video.global_fragments_path
    list_of_blobs = ListOfBlobs.load(blobs_path)
    blobs = list_of_blobs.blobs_in_video
    print("loading global fragments")
    global_fragments = np.load(global_fragments_path)

    # individual fragments statistics
    individual_fragments_added = []
    number_of_frames_in_individual_fragments = []
    distance_travelled_individual_fragments = []
    # global fragments statistics
    number_of_frames_in_longest_individual_fragment = [] #longest in terms of frames
    number_of_frames_in_shortest_individual_fragment = [] # shortest in terms of frames
    median_number_of_frames = []
    # minimum_number_of_frames_in_shortest_distance_travelled_individual_fragment = []
    distance_travelled_by_longest_distance_travelled_individual_fragment = []
    distance_travelled_by_shortes_distance_travelled_individual_fragment = []
    min_distance_travelled = []
    number_of_portraits_per_individual_fragment = []
    for global_fragment in global_fragments:
        # number_of_frames
        number_of_portraits_per_individual_fragment.append(global_fragment._number_of_portraits_per_individual_fragment)
        # maximum number of frames in global fragment
        number_of_frames_in_longest_individual_fragment.append(np.max(global_fragment._number_of_portraits_per_individual_fragment))
        # minimum number of images in global fragment
        number_of_frames_in_shortest_individual_fragment.append(np.min(global_fragment._number_of_portraits_per_individual_fragment))
        median_number_of_frames.append(np.median(global_fragment._number_of_portraits_per_individual_fragment))
        # compute minimum_distance_travelled for every blob in the individual fragment
        distance_travelled = [blob.distance_travelled_in_fragment()
                                        for blob in blobs[global_fragment.index_beginning_of_fragment]]
        min_distance_travelled.append(np.min(distance_travelled))
        # maximum distance travelled in global fragment
        distance_travelled_by_longest_distance_travelled_individual_fragment.append(np.max(distance_travelled))
        # minimum distance travelled in global fragment
        distance_travelled_by_shortes_distance_travelled_individual_fragment.append(np.min(distance_travelled))
        # number of images for the minimum distance travelled global fragment
        # index = np.argsort(distance_travelled)[0]
        # minimum_number_of_frames_in_shortest_distance_travelled_individual_fragment.append(number_of_frames_in_shortest_individual_fragment[index])

        for i, individual_fragment_identifier in enumerate(global_fragment.individual_fragments_identifiers):
            if individual_fragment_identifier not in individual_fragments_added:

                individual_fragments_added.append(individual_fragment_identifier)
                number_of_frames_in_individual_fragments.append(global_fragment._number_of_portraits_per_individual_fragment[i])
                distance_travelled_individual_fragments.append(distance_travelled[i])

    ''' plotting '''
    plt.ion()
    sns.set_style("ticks")
    fig, ax_arr = plt.subplots(2,4)
    plt.subplots_adjust(hspace = .3, wspace = .5)

    number_of_frames_in_individual_fragments_0 = filter(lambda x: x != 0, number_of_frames_in_individual_fragments)
    number_of_frames_in_shortest_individual_fragment_0 = filter(lambda x: x != 0, number_of_frames_in_shortest_individual_fragment)
    distance_travelled_individual_fragments_0 = filter(lambda x: x != 0, distance_travelled_individual_fragments)
    # number of frames in individual fragments
    nbins = 25
    ax = ax_arr[0,0]
    MIN = np.min(number_of_frames_in_individual_fragments_0)
    MAX = np.max(number_of_frames_in_individual_fragments_0)
    hist, bin_edges = np.histogram(number_of_frames_in_individual_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1], hist, '-ob' ,markersize = 5)
    # ax.plot(bin_edges[:-1], hist, '-ob' ,markersize = 5)
    ax.set_xlabel('num frames')
    ax.set_ylabel('num indiv fragments')

    # number of frames in shortest individual fragment
    ax = ax_arr[0,1]
    MIN = np.min(number_of_frames_in_shortest_individual_fragment_0)
    MAX = np.max(number_of_frames_in_shortest_individual_fragment_0)
    hist, bin_edges = np.histogram(number_of_frames_in_shortest_individual_fragment, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1],hist, 'ro-', markersize = 5)
    # ax.plot(bin_edges[:-1],hist, 'ro-', markersize = 5)
    ax.text(.5,.95,'only individual fragments \nwith minimum \nnumber of frames \nin global fragment',
        horizontalalignment='center',
        transform=ax.transAxes,
        verticalalignment = 'top')
    ax.set_xlabel('num frames')

    # distance travelled in individual fragments
    ax = ax_arr[0,2]
    MIN = np.min(distance_travelled_individual_fragments_0)
    MAX = np.max(distance_travelled_individual_fragments_0)
    hist, bin_edges = np.histogram(distance_travelled_individual_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1], hist, '-ob' ,markersize = 5)
    # ax.plot(bin_edges[:-1], hist, '-ob' ,markersize = 5)
    ax.set_xlabel('distance travelled (pixels)')

    # number of frames vs distance travelled
    ax = ax_arr[0,3]
    ax.plot(number_of_frames_in_individual_fragments, distance_travelled_individual_fragments, 'bo', alpha = .1, label = 'individual fragment', markersize = 5)
    ax.set_xlabel('num frames')
    ax.set_ylabel('distance travelled (pixels)')
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')


    ax = plt.subplot2grid((2, 4), (1, 0), colspan=4)
    index_order_by_max_num_frames = np.argsort(min_distance_travelled)[::-1]
    number_of_frames_in_longest_individual_fragment_ordered = np.asarray(number_of_frames_in_longest_individual_fragment)[index_order_by_max_num_frames]
    number_of_frames_in_shortest_individual_fragment_ordered = np.asarray(number_of_frames_in_shortest_individual_fragment)[index_order_by_max_num_frames]
    number_of_portraits_per_individual_fragment_ordered = np.asarray(number_of_portraits_per_individual_fragment)[index_order_by_max_num_frames]
    median_number_of_frames_ordered = np.asarray(median_number_of_frames)[index_order_by_max_num_frames]

    # ax.semilogy(range(len(global_fragments)), number_of_frames_in_longest_individual_fragment_ordered, color = 'r', linewidth= 2 ,alpha = .5)
    a = ax.semilogy(range(len(global_fragments)), median_number_of_frames_ordered, color = 'b', linewidth= 2, label = 'median')
    # ax.semilogy(range(len(global_fragments)), number_of_frames_in_shortest_individual_fragment_ordered, color = 'r', linewidth= 2 ,alpha = .5)
    for i in range(len(global_fragments)):
        a = ax.semilogy(i*np.ones(video.number_of_animals),number_of_portraits_per_individual_fragment_ordered[i],'o',alpha = .05,color = 'b',markersize=5,label='individual fragment')
    b = ax.semilogy(range(len(global_fragments)), number_of_frames_in_longest_individual_fragment_ordered, color = 'r', linewidth= 2 ,alpha = .5, label = 'max')
    c = ax.semilogy(range(len(global_fragments)), median_number_of_frames_ordered, color = 'r', linewidth= 2, label = 'median')
    d = ax.semilogy(range(len(global_fragments)), number_of_frames_in_shortest_individual_fragment_ordered, color = 'r', linewidth= 2 ,alpha = .5, label = 'min')
    ax.set_xlabel('global fragments ordered by minimum distance travelled (from max to min)')
    ax.set_ylabel('num of frames')
    ax.legend(handles = [c[0],d[0],b[0],a[0]])

    # # ax.semilogy(range(len(global_fragments)), number_of_frames_in_longest_individual_fragment_ordered, color = 'r', linewidth= 2 ,alpha = .5)
    # a = ax.plot(range(len(global_fragments)), median_number_of_frames_ordered, color = 'b', linewidth= 2, label = 'median')
    # # ax.semilogy(range(len(global_fragments)), number_of_frames_in_shortest_individual_fragment_ordered, color = 'r', linewidth= 2 ,alpha = .5)
    # for i in range(len(global_fragments)):
    #     a = ax.plot(i*np.ones(video.number_of_animals),number_of_portraits_per_individual_fragment_ordered[i],'o',alpha = .05,color = 'b',markersize=5,label='individual fragment')
    # b = ax.plot(range(len(global_fragments)), number_of_frames_in_longest_individual_fragment_ordered, color = 'r', linewidth= 2 ,alpha = .5, label = 'max')
    # c = ax.plot(range(len(global_fragments)), median_number_of_frames_ordered, color = 'r', linewidth= 2, label = 'median')
    # d = ax.plot(range(len(global_fragments)), number_of_frames_in_shortest_individual_fragment_ordered, color = 'r', linewidth= 2 ,alpha = .5, label = 'min')
    # ax.set_xlabel('global fragments ordered by minimum distance travelled (from max to min)')
    # ax.set_ylabel('num of frames')
    # ax.legend(handles = [c[0],d[0],b[0],a[0]])



    plt.show()
