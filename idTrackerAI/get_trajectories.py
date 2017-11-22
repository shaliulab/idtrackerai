"""
Usage: get_trajectories.py

Contains tools to process tracked ListOfBlobs
and output trajectories as numpy files with dimensions:

    [Individual number  x  frame number  x  coordinate (x,y)]

When a certain individual was not identified in the frame
a NaN appears instead of the coordinates
"""
import os
import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
# Import application/library specifics
sys.path.append('IdTrackerDeep/restructure')
sys.path.append('IdTrackerDeep/tracker')
sys.path.append('IdTrackerDeep/utils')
from tqdm import tqdm
import logging

from list_of_blobs import ListOfBlobs
from GUI_utils import selectDir

logger = logging.getLogger("__main__.get_trajectories")

def smooth_trajectories(t, sigma = 1.5, truncate = 4.0, derivative = 0):
    """Smooth trajectories (and maybe perform derivatives)

    :param t: trajectories as np.array [Individual, frame, coordinates]
    :param sigma: standard deviation of Gaussian kernel
    :param truncate: truncate filter at this number of std. Increase this if performing derivatives
    :param derivative: order of performed derivative (0: none, 1: first derivative ...)
    :returns: smoothed (and maybe derived) trajectoroes as np.array
    """
    t = gaussian_filter1d(t, sigma=sigma, axis=1, truncate = truncate, order = derivative)
    return t

def assign_point_to_identity(centroid, identity, frame_number, centroid_trajectories):
    if identity is not None and identity != 0:
        centroid_trajectories[frame_number, identity - 1, :] = centroid
    return centroid_trajectories

def produce_trajectories(blobs_in_video, number_of_frames, number_of_animals):
    """Produce trajectories from ListOfBlobs
    :param blob_file: ListOfBlobs instance
    :returns: A dictionary with np.array as values
    """
    centroid_trajectories = np.ones((number_of_frames, number_of_animals, 2))*np.NaN

    for frame_number, blobs_in_frame in enumerate(tqdm(blobs_in_video)):

        for blob in blobs_in_frame:

            if isinstance(blob.final_identity, int):
                centroid_trajectories = assign_point_to_identity(blob.centroid,
                                                                blob.final_identity,
                                                                blob.frame_number,
                                                                centroid_trajectories)
            elif isinstance(blob.final_identity, list):
                for identity, centroid in zip(blob.final_identity, blob.interpolated_centroids):
                    centroid_trajectories = assign_point_to_identity(centroid,
                                                                    identity,
                                                                    blob.frame_number,
                                                                    centroid_trajectories)

    return {"centroid": centroid_trajectories}

def save_trajectories(trajectories, trajectories_folder):
    """
    params:
        trajectories: dictionary of trajectories
        trajectories_folder: path to save trajectories
    """
    for name in trajectories:
        np.save(os.path.join(trajectories_folder, name + '_trajectories.npy'), trajectories[name])
        np.save(os.path.join(trajectories_folder, name + '_smooth_trajectories.npy'), smooth_trajectories(trajectories[name]))
        np.save(os.path.join(trajectories_folder, name + '_smooth_velocities.npy'), smooth_trajectories(trajectories[name], derivative = 1))
        np.save(os.path.join(trajectories_folder, name + '_smooth_accelerations.npy'), smooth_trajectories(trajectories[name], derivative = 2))

if __name__ == "__main__":
    # #SIMPLE USAGE EXAMPLE
    # BLOB_FILE_NAME = "blobs_collection.npy"
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)
    blobs_list = ListOfBlobs.load(video.blobs_path)

    trajectories_folder = os.path.join(video.session_folder,'trajectories')
    if not os.path.isdir(trajectories_folder):
        print("Creating trajectories folder...")
        os.makedirs(trajectories_folder)

    trajectories = produce_trajectories(blobs_list.blobs_in_video, video.number_of_frames, video.number_of_animals)
    save_trajectories(trajectories, trajectories_folder)
