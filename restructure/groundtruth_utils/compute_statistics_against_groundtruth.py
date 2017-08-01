from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np

"""Given two list of blobs, one deduced from human groundtruth and the other
generated by the tracking algorithm, compares them and gives back some statistics

Crossing: crossings are a special case. We ...
"""
def check_crossing_assignment(groundtruth_blob, tracked_blob, misassigned_blob_count_in_crossings):
    if tracked_blob.centroid not in groundtruth_blob.pixels:
        misassigned_blob_count_in_crossings[groundtruth_blob.identity] += 1
    else:
        if groundtruth_blob.identity != tracked_blob.identity:
            misassigned_blob_count_in_crossings[groundtruth_blob.identity] += 1
    return misassigned_blob_count_in_crossings

def compare_tracking_against_groundtruth(number_of_animals, blobs_list_groundtruth, blobs_list_tracked):
    misassigned_blob_count = {i: 0 for i in range(1, number_of_animals + 1)}
    misassigned_blob_count_in_crossings = {i: 0 for i in range(1, number_of_animals + 1)}

    for i, groundtruth_blobs_in_frame in enumerate(blobs_list_groundtruth):
        tracked_blobs_in_frame = blobs_list_tracked[i]

        for j, groundtruth_blob in enumerate(groundtruth_blobs_in_frame):
            tracked_blob = tracked_blobs_in_frame[j]
            if not groundtruth_blob.is_a_crossing:
                if groundtruth_blob.identity != tracked_blob.identity:
                    misassigned_blob_count[groundtruth_blob.identity] += 1
            else:
                misassigned_blob_count_in_crossings = check_crossing_assignment(groundtruth_blob, tracked_blob. misassigned_blob_count_in_crossings)
    return misassigned_blob_count, misassigned_blob_count_in_crossings

def get_statistics_against_groundtruth(groundtruth, blobs_list_tracked):
    number_of_animals = groundtruth.video_object.number_of_animals
    blobs_list_groundtruth = groundtruth.list_of_blobs

    misassigned_blob_count, misassigned_blob_count_in_crossings = compare_tracking_against_groundtruth(number_of_animals, blobs_list_groundtruth, blobs_list_tracked)

    unoccluded_individual_accuracy = {i :
                                    1 - misassigned_blob_count[i] / groundtruth.unoccluded_individual_assignments[i]
                                    for i in range(1, number_of_animals + 1)}
    unoccluded_accuracy = np.mean(unoccluded_individual_accuracy.values())

    individual_accuracy = {i :
                        1 - (misassigned_blob_count[i] + misassigned_blob_count_in_crossings[i])/ groundtruth.crossing_individual_assignments[i]
                        for i in range(1, number_of_animals + 1)}

    accuracy = np.mean(individual_accuracy.values())
    return unoccluded_individual_accuracy, unoccluded_accuracy, individual_accuracy, accuracy
