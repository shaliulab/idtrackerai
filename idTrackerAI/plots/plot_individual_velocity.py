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
from blob import Blob
from list_of_blobs import ListOfBlobs


def plot_individual_trajectories_velocities_and_accelerations(individual_trajectories):

    number_of_animals = individual_trajectories.shape[1]

    print('1')
    individual_velocities = np.diff(individual_trajectories, axis = 0)
    individual_velocities_magnitude = np.linalg.norm(individual_velocities, axis = 2)
    min_vel, max_vel = np.nanmin(individual_velocities_magnitude), np.nanmax(individual_velocities_magnitude)
    print('2')
    individual_accelerations = np.diff(individual_velocities, axis = 0)
    individual_accelerations_magnitude = np.linalg.norm(individual_accelerations, axis = 2)
    min_acc, max_acc = np.nanmin(individual_accelerations_magnitude), np.nanmax(individual_accelerations_magnitude)
    print('3')
    plt.ion()
    sns.set_style("ticks")
    fig = plt.figure()
    colors = get_spaced_colors_util(number_of_animals, norm=True, black=False)
    print('4')


    ''' X position '''
    ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=2)
    for i in range(number_of_animals):
        ax1.plot(individual_trajectories[:,i,0], color = colors[i])

    ''' Y position '''
    ax = plt.subplot2grid((3, 5), (0, 2), colspan=2, sharex=ax1)
    for i in range(number_of_animals):
        ax.plot(individual_trajectories[:,i,1], color = colors[i])


    ''' X-Y position '''
    ax = plt.subplot2grid((3, 5), (0, 4), colspan=1)
    for i in range(number_of_animals):
        ax.plot(individual_trajectories[:,i,0], individual_trajectories[:,i,1], color = colors[i])

    ''' individual velocities '''
    ax = plt.subplot2grid((3, 5), (1, 0), colspan=4, sharex=ax1)
    for i in range(number_of_animals):
        ax.plot(individual_velocities_magnitude[:,i], color = colors[i], label = str(i+1))
    ax.legend()

    ''' individual velocities distribution '''
    ax = plt.subplot2grid((3, 5), (1, 4), colspan=4)
    nbins = 100
    for i in range(number_of_animals):
        keep = ~np.isnan(individual_velocities_magnitude[:,i])
        # hist, bin_edges = np.histogram(individual_velocities_magnitude[i,keep], bins = 10**np.linspace(np.log10(min_vel),np.log10(max_vel),nbins))
        hist, bin_edges = np.histogram(individual_velocities_magnitude[keep,i], bins = np.linspace(min_vel,max_vel,nbins))
        ax.plot(bin_edges[:-1], hist, color = colors[i])

    ''' individual accelerations '''
    ax = plt.subplot2grid((3, 5), (2, 0), colspan=4, sharex=ax1)
    for i in range(number_of_animals):
        ax.plot(individual_accelerations_magnitude[:,i], color = colors[i])

    ''' individual accelerations distribution '''
    ax = plt.subplot2grid((3, 5), (2, 4), colspan=4)
    nbins = 100
    for i in range(number_of_animals):
        keep = ~np.isnan(individual_accelerations_magnitude[:,i])
        # hist, bin_edges = np.histogram(individual_accelerations_magnitude[i,keep], bins = 10**np.linspace(np.log10(min_vel),np.log10(max_vel),nbins))
        hist, bin_edges = np.histogram(individual_accelerations_magnitude[keep,i], bins = np.linspace(min_vel,max_vel,nbins))
        ax.plot(bin_edges[:-1], hist, color = colors[i])

    plt.show()


if __name__ == '__main__':
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)
    # blobs_path = video.blobs_path
    # list_of_blobs = ListOfBlobs.load(blobs_path)
    # blobs = list_of_blobs.blobs_in_video
    try:
        trajectories_dict = np.load(os.path.join(video.trajectories_wo_gaps_folder,'trajectories_wo_gaps.npy')).item()
        # trajectories_dict = np.load(os.path.join(video.trajectories_folder,'trajectories.npy')).item()
    except:
        trajectories_folder = selectDir("./")
        individual_trajectories = np.load(os.path.join(trajectories_folder,'centroid_trajectories.npy'))

    print('entering function')
    plot_individual_trajectories_velocities_and_accelerations(trajectories_dict['trajectories'])
