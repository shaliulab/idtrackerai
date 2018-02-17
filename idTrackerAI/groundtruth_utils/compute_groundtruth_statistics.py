from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
from pprint import pprint

from GUI_utils import selectDir
from list_of_blobs import ListOfBlobs
from blob import Blob
from generate_groundtruth import GroundTruth, GroundTruthBlob

"""Given two list of blobs_in_video, one deduced from human groundtruth and the other
generated by the tracking algorithm, compares them and gives back some statistics

Crossing: crossings are a special case. We ...
"""

if sys.argv[0] == 'idtrackeraiApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.compute_statistics_against_groundtruth")

def compare_tracking_against_groundtruth_no_gaps(number_of_animals,
                                                groundtruth, blobs_in_video_groundtruth,
                                                blobs_in_video, identities_dictionary_permutation):
    """
    blobs_in_video_groundtruth: cut groundtruth.blobs_in_video using start and end of the groundtruth object
    blobs_in_video: cut list_of_blobs.blobs_in_video using start and end of the groundtruth object
    """
    results = {}
    results['number_of_blobs_per_identity'] = {identity: 0 for identity in range(1, number_of_animals + 1)}
    results['number_of_individuals_badly_assigned'] = {identity: 0 for identity in range(1, number_of_animals + 1)}
    if identities_dictionary_permutation is not None:
        results['number_of_individuals_badly_interpolated'] = {
            identities_dictionary_permutation[identity]: groundtruth.wrong_crossing_counter[identity]
            for identity in groundtruth.wrong_crossing_counter}
        results['number_of_individuals_unidentified'] = {
            identities_dictionary_permutation[identity]: groundtruth.unidentified_individuals_counter[identity]
            for identity in groundtruth.wrong_crossing_counter}
    else:
        results['number_of_individuals_badly_interpolated'] = groundtruth.wrong_crossing_counter
        results['number_of_individuals_unidentified'] = groundtruth.unidentified_individuals_counter

    for blobs_in_frame_gt, blobs_in_frame in zip(blobs_in_video_groundtruth, blobs_in_video):
        for blob_gt, blob in zip(blobs_in_frame_gt, blobs_in_frame):

            if identities_dictionary_permutation is not None:
                if isinstance(blob_gt.identity, int):
                    gt_identity = identities_dictionary_permutation[blob_gt.identity]
                elif isinstance(blob_gt.identity, list):
                    gt_identity = [identities_dictionary_permutation[identity]
                                    for identity in blob_gt.identity]
            else:
                gt_identity = blob_gt.identity

            if isinstance(gt_identity, int) and gt_identity != 0:
                results['number_of_blobs_per_identity'][gt_identity] += 1
            elif isinstance(gt_identity, list):
                for identity in gt_identity:
                    if identity != 0:
                        results['number_of_blobs_per_identity'][identity] += 1
            elif gt_identity is None:
                logger.debug('***************************************unidentified blobs')

            if blob_gt.is_an_individual and not blob_gt.was_a_crossing:
                if gt_identity != blob.assigned_identity:
                    results['number_of_individuals_badly_assigned'][gt_identity] += 1

    results['number_of_errors_in_all_blobs'] = {i: results['number_of_individuals_badly_assigned'][i] +
                                                results['number_of_individuals_badly_interpolated'][i]
                                                for i in range(1, number_of_animals + 1)}

    return results

def get_accuracy_wrt_groundtruth_no_gaps(video, groundtruth,
                                            blobs_in_video_groundtruth,
                                            blobs_in_video = None,
                                            first_frame_first_global_fragment = None):

    check_ground_truth_consistency(blobs_in_video_groundtruth, blobs_in_video, first_frame_first_global_fragment)
    identities_dictionary_permutation = get_permutation_of_identities(first_frame_first_global_fragment,
                                                                    blobs_in_video_groundtruth, blobs_in_video)
    number_of_animals = video.number_of_animals
    if blobs_in_video is None:
        blobs_in_video = blobs_in_video_groundtruth
    results = compare_tracking_against_groundtruth_no_gaps(number_of_animals,
                                                    groundtruth, blobs_in_video_groundtruth,
                                                    blobs_in_video, identities_dictionary_permutation)

    accuracies = {}
    accuracies['individual_accuracy'] = {i : 1. - results['number_of_errors_in_all_blobs'][i] / results['number_of_blobs_per_identity'][i]
                            for i in range(1, number_of_animals + 1)}
    accuracies['accuracy'] = np.mean(accuracies['individual_accuracy'].values())
    logger.info(accuracies)
    logger.info(results)
    return accuracies, results

def compare_tracking_against_groundtruth(number_of_animals, blobs_in_video_groundtruth, blobs_in_video, identities_dictionary_permutation):
    #create dictionary to store eventual corrections made by the user
    results = {}
    results['number_of_blobs_per_identity'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['sum_individual_P2'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_assigned_blobs_per_identity'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_blobs_assigned_during_accumulation_per_identity'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_blobs_after_accumulation_per_identity'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_errors_in_all_blobs'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_errors_in_assigned_blobs'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_errors_in_blobs_assigned_during_accumulation'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_errors_in_blobs_after_accumulation'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_errors_in_blobs_assigned_after_accumulation'] = {i:0 for i in range(1, number_of_animals + 1)}
    results['number_of_individual_blobs'] = 0
    results['number_of_crossing_blobs'] = 0
    results['number_of_crossings_blobs_assigned_as_individuals'] = 0
    results['frames_with_identity_errors'] = []
    results['fragment_identifiers_with_identity_errors'] = []
    results['frames_with_crossing_errors'] = []
    results['fragment_identifiers_with_crossing_errors'] = []
    results['frames_with_zeros_in_groundtruth'] = []
    results['number_of_crossing_fragments'] = 0
    results['fragments_identifiers_of_crossings'] = []

    for groundtruth_blobs_in_frame, blobs_in_frame in zip(blobs_in_video_groundtruth, blobs_in_video):

        for groundtruth_blob, blob in zip(groundtruth_blobs_in_frame,blobs_in_frame):

            if identities_dictionary_permutation is not None:
                gt_identity = identities_dictionary_permutation[groundtruth_blob.identity]
            else:
                gt_identity = groundtruth_blob.identity

            if groundtruth_blob.is_an_individual and gt_identity != -1 and not groundtruth_blob.was_a_crossing: # we are not considering crossing or failures of the model area
                results['number_of_individual_blobs'] += 1
                if gt_identity == 0:
                    results['frames_with_zeros_in_groundtruth'].append(groundtruth_blob.frame_number)
                else:
                    try:
                        if blob.assigned_identity != 0 and blob.identity_corrected_closing_gaps is None: # we only consider P2 for non interpolated blobs
                            results['sum_individual_P2'][gt_identity] += blob._P2_vector[gt_identity - 1]
                    except:
                        logger.debug("P2_vector %s" %str(blob._P2_vector))
                        logger.debug("individual %s" %str(blob.is_an_individual))
                        logger.debug("fragment identifier ", blob.fragment_identifier)
                    results['number_of_blobs_per_identity'][gt_identity] += 1
                    results['number_of_assigned_blobs_per_identity'][gt_identity] += 1 if blob.assigned_identity != 0 else 0
                    results['number_of_blobs_assigned_during_accumulation_per_identity'][gt_identity] += 1 if blob.used_for_training else 0
                    results['number_of_blobs_after_accumulation_per_identity'][gt_identity] += 1 if not blob.used_for_training else 0
                    if gt_identity != blob.assigned_identity:
                        results['number_of_errors_in_all_blobs'][gt_identity] += 1
                        results['number_of_errors_in_blobs_after_accumulation'][gt_identity] += 1 if not blob.used_for_training else 0
                        if blob.assigned_identity != 0:
                            results['number_of_errors_in_assigned_blobs'][gt_identity] += 1
                            results['number_of_errors_in_blobs_assigned_during_accumulation'][gt_identity] += 1 if blob.used_for_training else 0
                            results['number_of_errors_in_blobs_assigned_after_accumulation'][gt_identity] += 1 if not blob.used_for_training else 0
                        if blob.fragment_identifier not in results['fragment_identifiers_with_identity_errors']:
                            results['frames_with_identity_errors'].append(blob.frame_number)
                            results['fragment_identifiers_with_identity_errors'].append(blob.fragment_identifier)

            elif groundtruth_blob.is_a_crossing or gt_identity == -1:
                if blob.fragment_identifier not in results['fragments_identifiers_of_crossings']:
                    results['fragments_identifiers_of_crossings'].append(blob.fragment_identifier)
                    results['number_of_crossing_fragments'] += 1
                results['number_of_crossing_blobs'] += 1
                results['number_of_crossings_blobs_assigned_as_individuals'] += 1 if blob.is_an_individual else 0
                if blob.is_an_individual:
                    if blob.fragment_identifier not in results['fragment_identifiers_with_crossing_errors']:
                        results['frames_with_crossing_errors'].append(blob.frame_number)
                        results['fragment_identifiers_with_crossing_errors'].append(blob.fragment_identifier)

    return results

def check_ground_truth_consistency(blobs_in_video_groundtruth, blobs_in_video, first_frame_first_global_fragment):

    if first_frame_first_global_fragment is not None \
        and first_frame_first_global_fragment > len(blobs_in_video_groundtruth):
        raise ValueError('The first_frame_first_global_fragment is bigger than the length of the groundtruth video')

    if len(blobs_in_video_groundtruth) != len(blobs_in_video):
        raise ValueError('Cannot compute the accuracy from a list of blobs with different length than the blobs_in_video_groundtruth')

    for blobs_in_frame_gt, blobs_in_frame in zip(blobs_in_video_groundtruth, blobs_in_video):

        if len(blobs_in_frame) != len(blobs_in_frame_gt):
            raise ValueError('Cannot compute the accuracy form a list of blobs with different blobs per frame than the blobs_in_video_groundtruth')

def get_permutation_of_identities(first_frame_first_global_fragment, blobs_in_video_groundtruth, blobs_in_video):
    if first_frame_first_global_fragment is not None:
        groundtruth_identities_in_first_frame = [blob.identity for blob in blobs_in_video_groundtruth[first_frame_first_global_fragment]]
        identities_in_first_frame = [blob.identity for blob in blobs_in_video[first_frame_first_global_fragment]]
        logger.debug('groundtruth identities in first frame %s' %str(groundtruth_identities_in_first_frame))
        logger.debug('identities in first frame %s' %str(identities_in_first_frame))

        identities_dictionary_permutation = {groundtruth_identity: identity
                                                for identity, groundtruth_identity in zip(identities_in_first_frame, groundtruth_identities_in_first_frame)}
    else:
        identities_dictionary_permutation = None

    return identities_dictionary_permutation

def get_accuracy_wrt_groundtruth(video, blobs_in_video_groundtruth, blobs_in_video = None, first_frame_first_global_fragment = None):
    check_ground_truth_consistency(blobs_in_video_groundtruth, blobs_in_video, first_frame_first_global_fragment)
    identities_dictionary_permutation = get_permutation_of_identities(first_frame_first_global_fragment, blobs_in_video_groundtruth, blobs_in_video)

    number_of_animals = video.number_of_animals
    if blobs_in_video is None:
        blobs_in_video = blobs_in_video_groundtruth
    results = compare_tracking_against_groundtruth(number_of_animals, blobs_in_video_groundtruth, blobs_in_video, identities_dictionary_permutation)
    if len(results['frames_with_zeros_in_groundtruth']) == 0:
        accuracies = {}
        accuracies['percentage_of_unoccluded_images'] = results['number_of_individual_blobs'] / (results['number_of_individual_blobs'] + results['number_of_crossing_blobs'])
        accuracies['individual_P2_in_validated_part'] = {i : results['sum_individual_P2'][i] / results['number_of_blobs_per_identity'][i]
                                for i in range(1, number_of_animals + 1)}
        accuracies['mean_individual_P2_in_validated_part'] = np.sum(results['sum_individual_P2'].values()) / np.sum(results['number_of_blobs_per_identity'].values())
        accuracies['individual_accuracy'] = {i : 1 - results['number_of_errors_in_all_blobs'][i] / results['number_of_blobs_per_identity'][i]
                                for i in range(1, number_of_animals + 1)}
        accuracies['accuracy'] = 1. - np.sum(results['number_of_errors_in_all_blobs'].values()) / np.sum(results['number_of_blobs_per_identity'].values())
        accuracies['individual_accuracy_assigned'] = {i : 1 - results['number_of_errors_in_assigned_blobs'][i] / results['number_of_assigned_blobs_per_identity'][i]
                                        for i in range(1, number_of_animals + 1)}
        accuracies['accuracy_assigned'] = 1. - np.sum(results['number_of_errors_in_assigned_blobs'].values()) / np.sum(results['number_of_assigned_blobs_per_identity'].values())
        accuracies['individual_accuracy_in_accumulation'] = {i : 1 - results['number_of_errors_in_blobs_assigned_during_accumulation'][i] / results['number_of_blobs_assigned_during_accumulation_per_identity'][i]
                                for i in range(1, number_of_animals + 1)}
        accuracies['accuracy_in_accumulation'] = 1. - np.sum(results['number_of_errors_in_blobs_assigned_during_accumulation'].values()) / np.sum(results['number_of_blobs_assigned_during_accumulation_per_identity'].values())
        accuracies['individual_accuracy_after_accumulation'] = {}
        for i in range(1, number_of_animals + 1):
            if results['number_of_blobs_after_accumulation_per_identity'][i] != 0:
                accuracies['individual_accuracy_after_accumulation'][i] = 1 - results['number_of_errors_in_blobs_after_accumulation'][i] / results['number_of_blobs_after_accumulation_per_identity'][i]
            else:
                accuracies['individual_accuracy_after_accumulation'][i] = None
        if np.sum(results['number_of_blobs_after_accumulation_per_identity'].values()) != 0:
            accuracies['accuracy_after_accumulation'] = 1. - np.sum(results['number_of_errors_in_blobs_after_accumulation'].values()) / np.sum(results['number_of_blobs_after_accumulation_per_identity'].values())
        else:
            accuracies['accuracy_after_accumulation'] = None
        if results['number_of_crossing_blobs'] != 0:
            accuracies['crossing_detector_accuracy'] = 1. - results['number_of_crossings_blobs_assigned_as_individuals'] / results['number_of_crossing_blobs']
        else:
            accuracies['crossing_detector_accuracy'] = None
        logger.info("accuracies %s" %str(accuracies))
        logger.info("number of crossing fragments in ground truth interval: %i" %results['number_of_crossing_fragments'])
        logger.info("number of crossing blobs in ground truth interval: %i" %results['number_of_crossing_blobs'])
        return accuracies, results

    else:
        logger.info("there are fish with 0 identity in frame %s" %str(results['frames_with_zeros_in_groundtruth']))
        return None, results

def compute_and_save_session_accuracy_wrt_groundtruth(video, groundtruth_type = None):
    logger.info("loading list_of_blobs")
    if groundtruth_type == 'normal':
        list_of_blobs = ListOfBlobs.load(video, video.blobs_path)
    elif groundtruth_type == 'interpolated':
        list_of_blobs = ListOfBlobs.load(video, video.blobs_path_interpolated)
    elif groundtruth_type == 'no_gaps':
        list_of_blobs = ListOfBlobs.load(video, video.blobs_no_gaps_path)
    #select ground truth file
    logger.info("loading groundtruth")
    if groundtruth_type == 'normal' or groundtruth_type == 'interpolated':
        groundtruth_path = os.path.join(video.video_folder,'_groundtruth.npy')
    elif groundtruth_type == 'no_gaps':
        groundtruth_path = os.path.join(video.video_folder,'_groundtruth_with_crossing_identified.npy')
    groundtruth = np.load(groundtruth_path).item()
    blobs_in_video_groundtruth = groundtruth.blobs_in_video[groundtruth.start:groundtruth.end]
    blobs_in_video = list_of_blobs.blobs_in_video[groundtruth.start:groundtruth.end]
    logger.info("computing groundtruth")
    if groundtruth_type == 'normal' or groundtruth_type == 'interpolated':
        accuracies, results = get_accuracy_wrt_groundtruth(video, blobs_in_video_groundtruth, blobs_in_video)
    elif groundtruth_type == 'no_gaps':
        accuracies, results = get_accuracy_wrt_groundtruth_no_gaps(video, groundtruth, blobs_in_video_groundtruth, blobs_in_video)
    if accuracies is not None:
        logger.info("saving accuracies in video")
        video.gt_start_end = (groundtruth.start,groundtruth.end)
        if groundtruth_type == 'normal':
            video.gt_accuracy = accuracies
            video.gt_results = results
        elif groundtruth_type == 'interpolated':
            video.gt_accuracy_interpolated = accuracies
            video.gt_results_interpolated = results
        elif groundtruth_type == 'no_gaps':
            video.gt_accuracy_no_gaps = accuracies
            video.gt_results_no_gaps = results
        video.save()

if __name__ == '__main__':
    groundtruth_type = sys.argv[1]
    #select blobs_in_video list tracked to compare against ground truth
    session_path = selectDir('./') #select path to video
    video_object_path = os.path.join(session_path,'video_object.npy')
    logger.info("loading video object")
    video = np.load(video_object_path).item(0)
    compute_and_save_session_accuracy_wrt_groundtruth(video, groundtruth_type)
