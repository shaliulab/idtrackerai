from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
from GUI_utils import selectDir
from blob import ListOfBlobs, Blob
from generate_light_groundtruth_blob_list import GroundTruth, GroundTruthBlob

"""Given two list of blobs, one deduced from human groundtruth and the other
generated by the tracking algorithm, compares them and gives back some statistics

Crossing: crossings are a special case. We ...
"""
def compare_tracking_against_groundtruth(number_of_animals, blobs_list_groundtruth, blobs_list_tracked):
    #create dictionary to store eventual corrections made by the user
    count_errors_identities_dict_assigned = {i:0 for i in range(1, number_of_animals + 1)}
    count_errors_identities_dict_all = {i:0 for i in range(1, number_of_animals + 1)}
    count_crossings_corrected_by_network = 0
    total_wrongly_assigned_crossings = 0
    frames_with_errors = []
    for groundtruth_blobs_in_frame, tracked_blobs_in_frame in zip(blobs_list_groundtruth, blobs_list_tracked):

        for groundtruth_blob, tracked_blob in zip(groundtruth_blobs_in_frame,tracked_blobs_in_frame):
            if tracked_blob._identity_corrected_solving_duplication is not None:
                tracked_blob_identity = tracked_blob._identity_corrected_solving_duplication
            elif tracked_blob._identity_corrected_solving_duplication is None:
                tracked_blob_identity = tracked_blob.identity

            if (tracked_blob.is_a_fish_in_a_fragment or\
                tracked_blob.is_a_jump or\
                tracked_blob.is_a_jumping_fragment or\
                hasattr(tracked_blob,'is_an_extreme_of_individual_fragment')) and\
                groundtruth_blob.identity != -1: # we are not considering crossing or failures of the model area
                # print("gt blob id ", groundtruth_blob.identity)
                # print(tracked_blob.frame_number)
                if groundtruth_blob.identity != tracked_blob_identity:
                    count_errors_identities_dict_all[groundtruth_blob.identity] += 1
                    frames_with_errors.append(tracked_blob.frame_number)
                    if tracked_blob_identity != 0:
                        count_errors_identities_dict_assigned[groundtruth_blob.identity] += 1
            elif groundtruth_blob.identity == -1:
                print("frame number ", tracked_blob.frame_number)
                total_wrongly_assigned_crossings += 1
                if tracked_blob_identity == 0 or tracked_blob_identity is None:
                    print("corrected")
                    count_crossings_corrected_by_network += 1

    if total_wrongly_assigned_crossings != 0:
        return count_errors_identities_dict_assigned, count_errors_identities_dict_all, count_crossings_corrected_by_network/total_wrongly_assigned_crossings, frames_with_errors
    else:
        return count_errors_identities_dict_assigned, count_errors_identities_dict_all, 1, frames_with_errors


def get_statistics_against_groundtruth(groundtruth, blobs_list_tracked):
    number_of_animals = groundtruth.video_object.number_of_animals
    blobs_list_groundtruth = groundtruth.list_of_blobs

    count_errors_identities_dict_assigned, count_errors_identities_dict_all, accuracy_crossing_detector, frames_with_errors = compare_tracking_against_groundtruth(number_of_animals, blobs_list_groundtruth, blobs_list_tracked)

    individual_accuracy_assigned = {i : 1 - count_errors_identities_dict_assigned[i] / groundtruth.count_number_assignment_per_individual_assigned[i] for i in range(1, number_of_animals + 1)}
    accuracy_assigned = np.mean(individual_accuracy_assigned.values())
    individual_accuracy = {i : 1 - count_errors_identities_dict_all[i] / groundtruth.count_number_assignment_per_individual_all[i] for i in range(1, number_of_animals + 1)}
    accuracy = np.mean(individual_accuracy.values())
    print("count_errors_identities_dict_assigned, ", count_errors_identities_dict_assigned)
    print("count_errors_identities_dict_all, ", count_errors_identities_dict_all)
    print("count_number_assignment_per_individual_assigned, ", groundtruth.count_number_assignment_per_individual_assigned)
    print("count_number_assignment_per_individual_all, ", groundtruth.count_number_assignment_per_individual_all)
    print("accuracy_crossing_detector, ", accuracy_crossing_detector)
    print("individual_accuracy_assigned, ", individual_accuracy_assigned)
    print("accuracy_assigned, ", accuracy_assigned)
    print("individual_accuracy, ", individual_accuracy)
    print("accuracy, ", accuracy)
    print("frames with errors, ", frames_with_errors)
    return accuracy, individual_accuracy, accuracy_assigned, individual_accuracy_assigned

if __name__ == '__main__':

    ''' select blobs list tracked to compare against ground truth '''
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)
    #change this
    blobs_path = video.blobs_path
    global_fragments_path = video.global_fragments_path
    list_of_blobs = ListOfBlobs.load(blobs_path)
    blobs = list_of_blobs.blobs_in_video


    ''' select ground truth file '''
    groundtruth_path = os.path.join(video._video_folder,'_groundtruth.npy')
    groundtruth = np.load(groundtruth_path).item()
    groundtruth.list_of_blobs = groundtruth.list_of_blobs[groundtruth.start:groundtruth.end]
    blobs = blobs[groundtruth.start:groundtruth.end]

    accuracy, individual_accuracy, accuracy_assigned, individual_accuracy_assigned = get_statistics_against_groundtruth(groundtruth, blobs)
