"""
Usage: get_trajectories.py

Contains tools to process tracked ListOfBlobs
and output trajectories as numpy files with dimensions:
    
    [Individual number  x  frame number  x  coordinate (x,y)]

When a certain individual was not identified in the frame
a NaN appears instead of the coordinates
"""

import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

# Import application/library specifics
sys.path.append('IdTrackerDeep/restructure')
sys.path.append('IdTrackerDeep/tracker')

from blob import ListOfBlobs

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
    
    for frame_number, blobs_in_frame in enumerate(list_of_blobs.blobs_in_video):
        print(frame_number)
        for blob in blobs_in_frame:
            if (blob.identity is not None) and (blob.identity != 0): #If blob is not a jump and it is not a crossing
                centroid_trajectories[blob.identity-1, frame_number, :] = blob.centroid
                try:
                    head_trajectories[blob.identity-1, frame_number, :] = blob.head_coordinates
                    nose_trajectories[blob.identity-1, frame_number, :] = blob.nose_coordinates
                except: #For compatibility with previous versions where portrait is a tuple
                    print("Warning: you are using an old ListOfBlobs file")
                    head_trajectories[blob.identity-1, frame_number, :] = blob.portrait[2]
                    nose_trajectories[blob.identity-1, frame_number, :] = blob.portrait[1]
 
    return {"centroid": centroid_trajectories, "nose": nose_trajectories, "head": head_trajectories}

if __name__ == "__main__":
    #SIMPLE USAGE EXAMPLE     
    BLOB_FILE_NAME = "blobs_collection.npy"
    trajectories = produce_trajectories(BLOB_FILE_NAME)
    for name in trajectories:
        np.save(name + '_trajectories.npy', trajectories[name])
        np.save(name + '_smooth_trajectories.npy', smooth_trajectories(trajectories[name]))
        np.save(name + '_smooth_velocities.npy', smooth_trajectories(trajectories[name], derivative = 1))
        np.save(name + '_smooth_accelerations.npy', smooth_trajectories(trajectories[name], derivative = 2))

