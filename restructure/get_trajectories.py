#import os
#from os.path import isdir, isfile
import sys
#import glob
import numpy as np
#import cPickle as pickle

# Import third party libraries
#import cv2

# Import application/library specifics
sys.path.append('IdTrackerDeep/restructure')
sys.path.append('IdTrackerDeep/tracker')

from blob import ListOfBlobs

BLOB_FILE_NAME = "blobs_collection.npy"
TRAJECTORIES_FILE_NAME = "trajectories.npy"

def produce_trajectories(file_name):
    list_of_blobs = ListOfBlobs.load(file_name)
    number_of_frames = len(list_of_blobs.blobs_in_video) 
    number_of_animals = list_of_blobs.blobs_in_video[0][0].number_of_animals #TODO number_of_animals should be a member of list_of_blobs
    trajectories = np.ones((number_of_frames, number_of_animals, 2))*np.NaN
    for frame_number, blobs_in_frame in enumerate(list_of_blobs.blobs_in_video):
        print(frame_number)
        for blob in blobs_in_frame:
            if blob.identity is not None:
                trajectories[frame_number, blob.identity-1, :] = blob.centroid
    return trajectories

if __name__ == "__main__":
    trajectories = produce_trajectories(BLOB_FILE_NAME)
    #print("Size: ", trajectories.shape)
    #print(trajectories)
    np.save(TRAJECTORIES_FILE_NAME, trajectories)

