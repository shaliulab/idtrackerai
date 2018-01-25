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
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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

from video import Video
from blob import Blob
from list_of_blobs import ListOfBlobs
from fragment import Fragment
from list_of_fragments import ListOfFragments
from globalfragment import GlobalFragment
from list_of_global_fragments import ListOfGlobalFragments
from GUI_utils import selectDir
from py_utils import get_spaced_colors_util


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
    list_of_blobs = ListOfBlobs.load(video, video.blobs_path)
    blobs = list_of_blobs.blobs_in_video
    list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_global_fragments = ListOfGlobalFragments.load(video.global_fragments_path, list_of_fragments.fragments)
    global_fragments = list_of_global_fragments.global_fragments
    global_fragment = global_fragments[1]

    strats_ends_individual_fragments = np.asarray([fragment.start_end for fragment in global_fragment.individual_fragments])
    min_start_individual_fragments = np.min(strats_ends_individual_fragments[:,0])
    max_end_individual_fragments = np.max(strats_ends_individual_fragments[:,1])
    max_start_individual_fragments = np.max(strats_ends_individual_fragments[:,0])
    min_end_individual_fragments = np.min(strats_ends_individual_fragments[:,1])
    blob_indices_individial_fragments = np.asarray([(fragment.blob_hierarchy_in_starting_frame, fragment.identifier) for fragment in global_fragment.individual_fragments])

    plt.ion()
    ''' global fragment sketch'''
    fig, ax_arr = plt.subplots(4,1, sharex=True)
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
                if blob.final_identity == 0:
                    ax_arr[0].plot([i, i+1],[j + 1,blob_index_next], 'o-' ,c = 'k', markersize = 3)
                else:
                    ax_arr[0].plot([i, i+1],[j + 1,blob_index_next], 'o-' ,c = '.75', markersize = 3)

                # hightlight invidiaul fragments in globl fragment
                if blob._fragment_identifier in global_fragment.individual_fragments_identifiers:
                    ax_arr[1].plot([i, i+1],[j + 1, blob_index_next], 'o-' ,c = colors[blob.final_identity], markersize = 3)
                else:
                    if blob.final_identity == 0:
                        ax_arr[1].plot([i, i+1],[j + 1,blob_index_next], 'o-' ,c = 'k', markersize = 3)
                    else:
                        ax_arr[1].plot([i, i+1],[j + 1,blob_index_next], 'o-' ,c = '.75', markersize = 3)

                # Extact individual fragments from global fragmtn
                blob_index_next = blobs[i+1].index(next_blob) + 1
                if blob._fragment_identifier in global_fragment.individual_fragments_identifiers and next_blob.is_an_individual:
                    ax_arr[2].plot([i, i+1],[j + 1, blob_index_next], 'o-' ,c = colors[blob.final_identity], markersize = 3)

                # Unroll hierarchies
                if blob._fragment_identifier in global_fragment.individual_fragments_identifiers and next_blob.is_an_individual:
                    blob_index = blob_indices_individial_fragments[np.where(blob_indices_individial_fragments[:,1] == blob._fragment_identifier)[0],:][0][0] + 1
                    ax_arr[3].plot([i,i+1],[blob_index, blob_index], '-' ,c = colors[blob.final_identity], linewidth = 10, solid_capstyle ='butt')
                    # ax_arr[3].plot([i,i+1],[blob.final_identity, blob.final_identity], 'o-' ,c = colors[blob.final_identity], markersize = 3, solid_capstyle ='butt')
                    # imscatter(i, blob_index, -blob.image_for_identification, ax = ax_arr[3], zoom = .3)
                    # imscatter(i+1 ,blob_index, -blob.next[0].portrait, ax = ax_arr[3], zoom = .3)

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
    fig.savefig('8fish_global_fragment_explanation.pdf', transparent=True)


    ''' individual fragments sketch single'''
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig, ax = plt.subplots(1)
    fig.set_size_inches((screen_x/100,screen_y/1.5/100))
    colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=True)

    ax.add_patch(Rectangle((max_start_individual_fragments-.5, 0.5),min_end_individual_fragments-max_start_individual_fragments + 1,video.number_of_animals ,alpha=1, fc = '0.85'))

    for i in range(min_start_individual_fragments, max_end_individual_fragments):
        blobs_in_frame = blobs[i]
        for j, blob in enumerate(blobs_in_frame):
            next_blobs = blob.next
            for next_blob in next_blobs:

                # Unroll hierarchies
                if blob._fragment_identifier in global_fragment.individual_fragments_identifiers and next_blob.is_an_individual_in_a_fragment:
                    blob_index = blob_indices_individial_fragments[np.where(blob_indices_individial_fragments[:,1] == blob._fragment_identifier)[0],:][0][0] + 1
                    ax.plot([i,i+1],[blob.final_identity, blob.final_identity], 'o-' ,c = colors[blob.final_identity], markersize = 5, solid_capstyle ='butt',linewidth = 1)


    ax.set_ylabel('blob index')
    ax.set_xlabel('frame number')
    ax.set_yticks(list(range(1,video.number_of_animals+1)))
    ax.set_yticklabels(list(range(1,video.number_of_animals+1)))
    fig.savefig('8fish_global_fragment_portraits.pdf', transparent=True)

    ''' individual fragments sketch single with images'''
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig, ax = plt.subplots(1)
    fig.set_size_inches((screen_x/100,screen_y/1.5/100))
    colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=True)

    ax.add_patch(Rectangle((max_start_individual_fragments-.5, 0.5),min_end_individual_fragments-max_start_individual_fragments + 1,video.number_of_animals ,alpha=1, fc = '0.85'))

    for i in range(min_start_individual_fragments, max_end_individual_fragments):
        blobs_in_frame = blobs[i]
        for j, blob in enumerate(blobs_in_frame):
            next_blobs = blob.next
            for next_blob in next_blobs:

                # Unroll hierarchies
                if blob._fragment_identifier in global_fragment.individual_fragments_identifiers and next_blob.is_an_individual:
                    blob_index = blob_indices_individial_fragments[np.where(blob_indices_individial_fragments[:,1] == blob._fragment_identifier)[0],:][0][0] + 1
                    ax.plot([i,i+1],[blob.final_identity, blob.final_identity], '-' ,c = colors[blob.final_identity], linewidth = 20, solid_capstyle ='butt')
                    # imscatter(i, blob.final_identity, -blob.image_for_identification, ax = ax, zoom = 1)
                    # imscatter(i+1 ,blob.final_identity, -blob.next[0].portrait, ax = ax, zoom = 1    )

    ax.set_ylabel('blob index')
    ax.set_xlabel('frame number')
    ax.set_yticks(list(range(1,video.number_of_animals+1)))
    ax.set_yticklabels(list(range(1,video.number_of_animals+1)))
    frame_range = [740, 758]
    ax.set_xlim(frame_range)
    blob_index_example = 4
    ax.set_ylim([blob_index_example - .5, blob_index_example + .5])
    fig.savefig('8fish_global_fragment_portraits_zoom.pdf', transparent=True)


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
            if blob._fragment_identifier in global_fragment.individual_fragments_identifiers and blob.is_an_individual:
                centroid_trajectories[i, blob.final_identity-1, :] = blob.centroid

    for individual in range(video.number_of_animals):
        ax.plot(centroid_trajectories[:,individual,0], centroid_trajectories[:,individual,1], range(min_start_individual_fragments, max_end_individual_fragments), color = colors[individual + 1] )
        # ax.plot(centroid_trajectories[:,individual,0], np.ones(number_of_frames)*650, range(number_of_frames),color = colors[individual + 1], alpha = 1 , linewidth = 1)
        # ax.plot(np.ones(number_of_frames)*950, centroid_trajectories[:,individual,1], range(number_of_frames),color = colors[individual + 1], alpha = 1, linewidth = 1)

    ax.view_init(5, -135)
    print(screen_y, screen_x)
    fig.set_size_inches((screen_x/3/100,screen_y/100))

    import mpl_toolkits.mplot3d.art3d as art3d
    from matplotlib.patches import Circle, PathPatch
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    p1 = Rectangle((x_lim[0],max_start_individual_fragments-1),np.diff(x_lim),min_end_individual_fragments-max_start_individual_fragments,alpha=1, fc = '0.8')
    p2 = Rectangle((y_lim[0],max_start_individual_fragments-1),np.diff(y_lim),min_end_individual_fragments-max_start_individual_fragments,alpha=1, fc = '0.85')
    ax.add_patch(p1)
    ax.add_patch(p2)
    art3d.pathpatch_2d_to_3d(p1, z=y_lim[1], zdir="y")
    art3d.pathpatch_2d_to_3d(p2, z=x_lim[1], zdir="x")

    # Get rid of the ticks
    ax.set_yticks([])
    ax.set_xticks([])
    #ax.set_xlim((750,950))
    #ax.set_ylim((50,650))
    ax.set_zlabel('frame number', labelpad=20)
    ax.zaxis.set_rotate_label(True)
    fig.savefig('8fish_3dtrajectories.pdf', transparent=True)

    # ax.set_zlim([245-min_start_individual_fragments,314-min_start_individual_fragments])
