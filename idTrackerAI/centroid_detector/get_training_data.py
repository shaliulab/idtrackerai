from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
from tqdm import tqdm

from blob import ListOfBlobs

#load list of blobs
list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_12/preprocessing/blobs_collection.npy'
list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
blobs = list_of_blobs.blobs_in_video

training_labels = {i: [None] * len(blobs) for i in range(1, blobs[0][0].number_of_animals + 1)}

for frame_num, blobs_in_frame in tqdm(enumerate(blobs)):
    for blob in blobs_in_frame:
        if blob.identity is not 0:
            training_labels[blob.identity][frame_num] = blob.centroid
