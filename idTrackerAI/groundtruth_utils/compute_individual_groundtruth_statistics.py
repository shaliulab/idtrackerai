from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
import logging
from pprint import pprint

from GUI_utils import selectDir
from list_of_blobs import ListOfBlobs
from blob import Blob
from generate_individual_groundtruth import IndividualGroundTruth, GroundTruthBlob

logger = logging.getLogger("__main__.compute_statistics_against_groundtruth")

def compare_tracked_individual_against_groundtruth(blobs_in_individual_groundtruth,
                                                    individual_blobs, individual_groundtruth_id):
    comparison_keys = ['accuracy', 'frames_with_errors', 'mistaken_identities']
    comparison_info = {key: [] for key in comparison_keys}

    for blob_gt, blob in zip(blobs_in_individual_groundtruth, individual_blobs):
        if blob_gt.identity != blob.assigned_identity:
            comparison_info['frames_with_errors'].append(blob.frame_number)
            comparison_info['mistaken_identities'].append(blob.assigned_identity)

    comparison_info['accuracy'] = 1 -len(comparison_info['mistaken_identities']) / len(blobs_in_individual_groundtruth)
    number_of_assigned_blobs = len([blob for blob in blobs_in_individual_groundtruth if blob.assigned_identity != 0])
    number_of_mistaken_identified_blobs = len([identity for identity in comparison_info['mistaken_identities'] if identity != 0])
    comparison_info['accuracy_assigned'] = 1 - number_of_mistaken_identified_blobs / number_of_assigned_blobs
    comparison_info['id'] = individual_groundtruth_id
    return comparison_info

def check_groundtruth_consistency(blobs_in_individual_groundtruth,
                                    individual_groundtruth_id, individual_blobs,
                                    individual_id):
    non_matching_error = "The length of the collections of the ground truth individual and the selected one do not match"
    if individual_groundtruth_id != individual_id or\
        len(blobs_in_individual_groundtruth) != len(individual_blobs):
        raise ValueError(non_matching_error)

def get_individual_accuracy_wrt_groundtruth(video, blobs_in_individual_groundtruth, individual_blobs = None):
    individual_groundtruth_id = blobs_in_individual_groundtruth[0].identity
    if individual_blobs is None:
        individual_blobs = blobs_in_individual_groundtruth
        individual_id = individual_groundtruth_id
    else:
        check_groundtruth_consistency(blobs_in_individual_groundtruth,
                                    individual_groundtruth_id,
                                    individual_blobs,
                                    individual_id)
    return compare_tracked_individual_against_groundtruth(blobs_in_individual_groundtruth,
                                                individual_blobs, individual_groundtruth_id)
