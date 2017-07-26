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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

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

    global_fragment = global_fragments[2]

    global_fragment.compute_start_end_frame_indices_of_individual_fragments(blobs)
    strats_ends_individual_fragments = np.asarray(global_fragment.starts_ends_individual_fragments)
    print(strats_ends_individual_fragments)
    min_start_individual_fragments = np.min(strats_ends_individual_fragments[:,0])
    max_end_individual_fragments = np.max(strats_ends_individual_fragments[:,1])
    max_start_individual_fragments = np.max(strats_ends_individual_fragments[:,0])
    min_end_individual_fragments = np.min(strats_ends_individual_fragments[:,1])
    blob_indices_individial_fragments = np.asarray([(blob._blob_index,fragment_identifier) for blob, fragment_identifier in zip(blobs[global_fragment.index_beginning_of_fragment],global_fragment.individual_fragments_identifiers)])

    ''' 3d trajectories '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=True)

    centroid_trajectories = np.ones((video._num_frames, video.number_of_animals, 2))*np.NaN

    for frame_number in range(video._num_frames):
        blobs_in_frame = blobs[frame_number]
        for j, blob in enumerate(blobs_in_frame):
            if blob.is_a_fish:
                centroid_trajectories[frame_number, blob.identity-1, :] = blob.centroid

    for individual in range(video.number_of_animals):
        ax.plot(centroid_trajectories[:,individual,0], centroid_trajectories[:,individual,1], range(video._num_frames),color = colors[individual + 1] )
        # ax.plot(centroid_trajectories[:,individual,0], np.ones(number_of_frames)*650, range(number_of_frames),color = colors[individual + 1], alpha = 1 , linewidth = 1)
        # ax.plot(np.ones(number_of_frames)*950, centroid_trajectories[:,individual,1], range(number_of_frames),color = colors[individual + 1], alpha = 1, linewidth = 1)

    ax.view_init(5, -135)
    print(screen_y, screen_x)
    fig.set_size_inches((screen_x/3/100,screen_y/100))

    # import mpl_toolkits.mplot3d.art3d as art3d
    # from matplotlib.patches import Circle, PathPatch
    # p1 = Rectangle((750,max_start_individual_fragments-1),200,min_end_individual_fragments-max_start_individual_fragments,alpha=1, fc = '0.8')
    # p2 = Rectangle((50,max_start_individual_fragments-1),600,min_end_individual_fragments-max_start_individual_fragments,alpha=1, fc = '0.85')
    # ax.add_patch(p1)
    # ax.add_patch(p2)
    # art3d.pathpatch_2d_to_3d(p1, z=650, zdir="y")
    # art3d.pathpatch_2d_to_3d(p2, z=950, zdir="x")

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlim((750,950))
    # ax.set_ylim((50,650))
    ax.set_zlabel('frame number', labelpad=20)
    ax.zaxis.set_rotate_label(True)
    # fig.savefig('8fish_3dtrajectories.pdf', transparent=True)
    plt.show()

    # ax.set_zlim([245-min_start_individual_fragments,314-min_start_individual_fragments])
