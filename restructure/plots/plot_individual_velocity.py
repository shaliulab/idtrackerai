from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys
import numpy as np

# Import third party libraries
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./')

from video import Video
from GUI_utils import selectDir
from py_utils import get_spaced_colors_util


if __name__ == '__main__':
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)

    individual_velocities = np.load(os.path.join(video.trajectories_folder,'centroid_smooth_velocities.npy'))
    individual_velocities_magnitude = np.linalg.norm(individual_velocities, axis = 2)
    min_vel, max_vel = np.nanmin(individual_velocities_magnitude), np.nanmax(individual_velocities_magnitude)

    individual_accelerations = np.load(os.path.join(video.trajectories_folder,'centroid_smooth_accelerations.npy'))
    individual_accelerations_magnitude = np.linalg.norm(individual_accelerations, axis = 2)
    min_acc, max_acc = np.nanmin(individual_accelerations_magnitude), np.nanmax(individual_accelerations_magnitude)

    plt.ion()
    fig = plt.figure()
    colors = get_spaced_colors_util(video.number_of_animals, norm=True, black=False)

    ''' individual velocities '''
    ax = plt.subplot2grid((2, 5), (0, 0), colspan=4)
    for i in range(video.number_of_animals):
        ax.plot(individual_velocities_magnitude[i,:], color = colors[i])

    ''' individual velocities distribution '''
    ax = plt.subplot2grid((2, 5), (0, 4), colspan=4)
    nbins = 100
    for i in range(video.number_of_animals):
        keep = ~np.isnan(individual_velocities_magnitude[i,:])
        hist, bin_edges = np.histogram(individual_velocities_magnitude[i,keep], bins = 10**np.linspace(np.log10(min_vel),np.log10(max_vel),nbins))
        # hist, bin_edges = np.histogram(individual_velocities_magnitude[i,keep], bins = np.linspace(min_vel,max_vel,nbins))
        ax.plot(bin_edges[:-1], hist, color = colors[i])

    ''' individual accelerations '''
    ax = plt.subplot2grid((2, 5), (1, 0), colspan=4)
    for i in range(video.number_of_animals):
        ax.plot(individual_accelerations_magnitude[i,:], color = colors[i])

    ''' individual accelerations distribution '''
    ax = plt.subplot2grid((2, 5), (1, 4), colspan=4)
    nbins = 100
    for i in range(video.number_of_animals):
        keep = ~np.isnan(individual_accelerations_magnitude[i,:])
        hist, bin_edges = np.histogram(individual_accelerations_magnitude[i,keep], bins = 10**np.linspace(np.log10(min_vel),np.log10(max_vel),nbins))
        # hist, bin_edges = np.histogram(individual_accelerations_magnitude[i,keep], bins = np.linspace(min_vel,max_vel,nbins))
        ax.plot(bin_edges[:-1], hist, color = colors[i])

    plt.show()
