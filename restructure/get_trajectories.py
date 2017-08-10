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
from blob import ListOfBlobs
from GUI_utils import selectDir


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

def produce_trajectories(blob_file):
    """Produce trajectories from ListOfBlobs

    :param blob_file: ListOfBlobs instance
    :returns: A dictionary with np.array as values
    """
    list_of_blobs = ListOfBlobs.load(blob_file)
    number_of_frames = len(list_of_blobs.blobs_in_video)
    number_of_animals = list_of_blobs.blobs_in_video[0][0].number_of_animals #TODO number_of_animals should be a member of list_of_blobs

    centroid_trajectories = np.ones((number_of_animals,number_of_frames, 2))*np.NaN
    nose_trajectories = np.ones((number_of_animals,number_of_frames, 2))*np.NaN
    head_trajectories = np.ones((number_of_animals, number_of_frames, 2))*np.NaN

    missing_head = False
    for frame_number, blobs_in_frame in enumerate(tqdm(list_of_blobs.blobs_in_video)):
        for blob in blobs_in_frame:
            if (blob.identity is not None) and (blob.identity != 0): #If blob is not a jump and it is not a crossing
                centroid_trajectories[blob.identity-1, frame_number, :] = blob.centroid
                try:
                    head_trajectories[blob.identity-1, frame_number, :] = blob.head_coordinates
                    nose_trajectories[blob.identity-1, frame_number, :] = blob.nose_coordinates
                except:
                    missing_head = True
    if missing_head:
        return {"centroid": centroid_trajectories, "head": head_trajectories, "nose": nose_trajectories}
    return {"centroid": centroid_trajectories}

if __name__ == "__main__":
    # #SIMPLE USAGE EXAMPLE
    # BLOB_FILE_NAME = "blobs_collection.npy"
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)
    blobs_path = video.blobs_path
    trajectories_folder = os.path.join(video._session_folder,'trajectories')
    if not os.path.isdir(trajectories_folder):
        print("Creating trajectories folder...")
        os.makedirs(trajectories_folder)
    trajectories = produce_trajectories(blobs_path)
    for name in trajectories:
        np.save(os.path.join(trajectories_folder, name + '_trajectories.npy'), trajectories[name])
        np.save(os.path.join(trajectories_folder, name + '_smooth_trajectories.npy'), smooth_trajectories(trajectories[name]))
        np.save(os.path.join(trajectories_folder, name + '_smooth_velocities.npy'), smooth_trajectories(trajectories[name], derivative = 1))
        np.save(os.path.join(trajectories_folder,name + '_smooth_accelerations.npy'), smooth_trajectories(trajectories[name], derivative = 2))
