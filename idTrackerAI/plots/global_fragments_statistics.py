from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys
# Import application/library specifics
sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./')
# sys.path.append('IdTrackerDeep/tracker')

# Import third party libraries
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np

from video import Video
from list_of_fragments import ListOfFragments
from list_of_global_fragments import ListOfGlobalFragments
from GUI_utils import selectDir

""" plotter """
def compute_and_plot_fragments_statistics(video, list_of_fragments, list_of_global_fragments):

    number_of_images_in_individual_fragments, \
    distance_travelled_individual_fragments, \
    number_of_images_in_crossing_fragments =  list_of_fragments.get_data_plot()

    number_of_images_in_shortest_individual_fragment,\
    number_of_images_in_longest_individual_fragment,\
    number_of_images_per_individual_fragment_in_global_fragment,\
    median_number_of_images,\
    minimum_distance_travelled = list_of_global_fragments.get_data_plot()
    # print(number_of_images_in_individual_fragments, \
    # distance_travelled_individual_fragments, \
    # number_of_images_in_crossing_fragments)
    # print("--------------------------------------------------------")
    # print(number_of_images_in_shortest_individual_fragment,\
    # number_of_images_in_longest_individual_fragment,\
    # number_of_images_per_individual_fragment_in_global_fragment,\
    # median_number_of_images,\
    # minimum_distance_travelled)


    ''' plotting '''
    plt.ion()
    sns.set_style("ticks")
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig, ax_arr = plt.subplots(2,4)
    fig.canvas.set_window_title('Fragments summary')
    fig.set_size_inches((screen_x/100,screen_y/100))
    plt.subplots_adjust(hspace = .3, wspace = .5)
    # number of frames in individual fragments
    nbins = 25
    ax = ax_arr[0,0]
    MIN = np.min(number_of_images_in_individual_fragments)
    MAX = np.max(number_of_images_in_individual_fragments)
    hist, bin_edges = np.histogram(number_of_images_in_individual_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1], hist, '-ob' ,markersize = 5)
    ax.set_xlabel('number of frames')
    ax.set_ylabel('number of individual fragments')
    # distance travelled in individual fragments
    non_zero_indices = np.where(distance_travelled_individual_fragments != 0)[0]
    distance_travelled_individual_fragments_non_zero = distance_travelled_individual_fragments[non_zero_indices]
    ax = ax_arr[0,1]
    MIN = np.min(distance_travelled_individual_fragments_non_zero)
    MAX = np.max(distance_travelled_individual_fragments_non_zero)
    hist, bin_edges = np.histogram(distance_travelled_individual_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1], hist, '-ob' ,markersize = 5)
    ax.set_xlabel('distance travelled (pixels)')
    # number of frames vs distance travelled
    ax = ax_arr[0,2]
    ax.plot(np.asarray(number_of_images_in_individual_fragments)[non_zero_indices], distance_travelled_individual_fragments_non_zero, 'bo', alpha = .1, label = 'individual fragment', markersize = 5)
    ax.set_xlabel('num frames')
    ax.set_ylabel('distance travelled (pixels)')
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    # number of frames in shortest individual fragment
    ax = ax_arr[0,3]
    MIN = np.min(number_of_images_in_crossing_fragments)
    MAX = np.max(number_of_images_in_crossing_fragments)
    hist, bin_edges = np.histogram(number_of_images_in_crossing_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1],hist, 'ro-', markersize = 5)
    ax.set_xlabel('number of images')
    ax.set_ylabel('number of crossing fragments')
    # plot global fragments
    ax = plt.subplot2grid((2, 4), (1, 0), colspan=4)
    index_order_by_max_num_frames = np.argsort(minimum_distance_travelled)[::-1]
    number_of_images_in_longest_individual_fragment = np.asarray(number_of_images_in_longest_individual_fragment)[index_order_by_max_num_frames]
    number_of_images_in_shortest_individual_fragment = np.asarray(number_of_images_in_shortest_individual_fragment)[index_order_by_max_num_frames]
    number_of_images_per_individual_fragment_in_global_fragment = np.asarray(number_of_images_per_individual_fragment_in_global_fragment)[index_order_by_max_num_frames]
    median_number_of_images = np.asarray(median_number_of_images)[index_order_by_max_num_frames]
    a = ax.semilogy(range(list_of_global_fragments.number_of_global_fragments), median_number_of_images, color = 'b', linewidth= 2, label = 'median')

    for i in range(list_of_global_fragments.number_of_global_fragments):
        a = ax.semilogy(i*np.ones(video.number_of_animals),number_of_images_per_individual_fragment_in_global_fragment[i],'o',alpha = .05,color = 'b',markersize=5,label='individual fragment')
    b = ax.semilogy(range(list_of_global_fragments.number_of_global_fragments), number_of_images_in_longest_individual_fragment, color = 'r', linewidth= 2 ,alpha = .5, label = 'max')
    c = ax.semilogy(range(list_of_global_fragments.number_of_global_fragments), median_number_of_images, color = 'r', linewidth= 2, label = 'median')
    d = ax.semilogy(range(list_of_global_fragments.number_of_global_fragments), number_of_images_in_shortest_individual_fragment, color = 'r', linewidth= 2 ,alpha = .5, label = 'min')
    ax.set_xlabel('global fragments ordered by minimum distance travelled (from max to min)')
    ax.set_ylabel('num of frames')
    ax.legend(handles = [c[0],d[0],b[0],a[0]])

    plt.show()
    fig.savefig(os.path.join(video._preprocessing_folder,'global_fragments_summary.pdf'), transparent=True)
    return number_of_images_in_individual_fragments, distance_travelled_individual_fragments

if __name__ == '__main__':
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item(0)
    list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_global_fragments = ListOfGlobalFragments.load(video.global_fragments_path, list_of_fragments.fragments)
    compute_and_plot_fragments_statistics(video, list_of_fragments, list_of_global_fragments)
