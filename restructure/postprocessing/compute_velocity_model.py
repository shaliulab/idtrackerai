from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
import numpy as np
from tqdm import tqdm
import collections
from blob import ListOfBlobs
from blob import Blob

def compute_model_velocity(blobs_in_video, number_of_animals, percentile = None):
    """computes the 2 * (percentile) of the distribution of velocities of identified fish.
    params
    -----
    blobs_in_video: list of blob objects
        collection of blobs detected in the video.
    number_of_animals int
    percentile int
    -----
    return
    -----
    float
    2 * np.max(distance_travelled_in_individual_fragments) if percentile is None
    2 * percentile(velocity distribution of identified animals) otherwise
    """
    distance_travelled_in_individual_fragments = []
    current_individual_fragment_identifier = -1

    for blobs_in_frame in tqdm( blobs_in_video, desc = "computing velocity model"):

        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment and current_individual_fragment_identifier != blob.fragment_identifier:
                current_individual_fragment_identifier = blob.fragment_identifier
                distance_travelled_in_individual_fragments.extend(blob.frame_by_frame_velocity())

    return 2 * np.max(distance_travelled_in_individual_fragments) if percentile is None else 2 * np.percentile(distance_travelled_in_individual_fragments, percentile)

def compute_velocity_from_list_of_blobs(list_of_blobs):
    centroids = [blob.centroid for blob in list_of_blobs]
    print(centroids)
    velocity = [np.linalg.norm(centroids[i+1] - centroid) for i, centroid in enumerate(centroids[:-1])]
    print(velocity)
    return np.mean(velocity)
