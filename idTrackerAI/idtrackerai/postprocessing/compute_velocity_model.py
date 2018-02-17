from __future__ import absolute_import, print_function, division
import sys
import numpy as np
from tqdm import tqdm
import collections
from idtrackerai.list_of_fragments import ListOfFragments

def compute_model_velocity(fragments, number_of_animals, percentile = None):
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

    for fragment in tqdm(fragments, desc = "computing velocity model"):
        if fragment.is_an_individual:
            distance_travelled_in_individual_fragments.extend(fragment.frame_by_frame_velocity())

    return 2 * np.max(distance_travelled_in_individual_fragments) if percentile is None else 2 * np.percentile(distance_travelled_in_individual_fragments, percentile)

def compute_velocity_from_list_of_fragments(list_of_fragments):
    return np.mean([fragment.frame_by_frame_velocity() for fragment in list_of_fragments])
