from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from network_params import NetworkParams
from get_data import DataSet
from id_CNN import ConvNetwork
from get_predictions import GetPrediction
from blob import get_images_from_blobs_in_video
from visualize_embeddings import EmbeddingVisualiser
from globalfragment import get_images_and_labels_from_global_fragment
from statistics_for_assignment import compute_P2_of_individual_fragment_from_blob, compute_P1_individual_fragment_from_blob, compute_identification_frequencies_individual_fragment

def assign(net, video, images, print_flag):
    # build data object
    images = np.expand_dims(np.asarray(images), axis = 3)
    data = DataSet(net.params.number_of_animals, images)
    # Instantiate data_set
    data.standarize_images()
    # Crop images from 36x36 to 32x32 without performing data augmentation
    data.crop_images(image_size = 32)
    # Train network
    assigner = GetPrediction(data, print_flag = print_flag)
    assigner.get_predictions_softmax(net.predict)
    # assigner.get_predictions_fully_connected_embedding(net.get_fully_connected_vectors, video.number_of_animals)
    # video.create_embeddings_folder()
    # visualize_fully_connected_embedding = EmbeddingVisualiser(labels = assigner._predictions,
    #                                                         features = assigner._fc_vectors)
    # visualize_fully_connected_embedding.create_labels_file(video._embeddings_folder)
    # visualize_fully_connected_embedding.visualize(video._embeddings_folder)
    return assigner

# def assign_identity_to_blobs_in_video(blobs_in_video, assigner):
#     counter = 0
#     for frame in blobs_in_video:
#         for blob in frame:
#             if not blob.is_a_fish_in_a_fragment:
#                 blob._identity = 0
#             elif blob._identity == None:
#                 blob._identity = int(assigner._predictions[counter])
#                 counter += 1
def assign_identity_to_blobs_in_video(blobs_in_video, assigner):
    counter = 0
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment and not blob.assigned_during_accumulation:
                assert blob._identity == None
                blob._identity = int(assigner._predictions[counter])
                counter += 1
            elif not blob.is_a_fish_in_a_fragment:
                blob._identity = 0

# def compute_P1_for_blobs_in_video(video, blobs_in_video):
#     """Assigns individual-fragment-based identities to all the blobs
#     in the video. It uses P1
#     """
#     for i, frame in enumerate(blobs_in_video):
#         for j, blob in enumerate(frame):
#             if blob.is_a_fish_in_a_fragment and not blob.previous[0].is_a_fish_in_a_fragment and not blob.assigned_during_accumulation:
#                 identities_in_fragment = np.asarray(blob.identities_in_fragment())
#                 frequencies_in_fragment = compute_identification_frequencies_individual_fragment(identities_in_fragment, video.number_of_animals)
#                 blob._frequencies_in_fragment = frequencies_in_fragment
#                 blob._P1_vector = compute_P1_individual_fragment_from_blob(frequencies_in_fragment)
#                 blob.update_P1_in_fragment()

def compute_P1_for_blobs_in_video(video, blobs_in_video):
    """Assigns individual-fragment-based identities to all the blobs
    in the video. It uses P1
    """
    for i, frame in enumerate(blobs_in_video):
        for j, blob in enumerate(frame):
            if blob.is_a_fish_in_a_fragment and not blob.assigned_during_accumulation:
                identities_in_fragment = np.asarray(blob.identities_in_fragment())
                frequencies_in_fragment = compute_identification_frequencies_individual_fragment(identities_in_fragment, video.number_of_animals)
                blob._frequencies_in_fragment = frequencies_in_fragment
                blob._P1_vector = compute_P1_individual_fragment_from_blob(frequencies_in_fragment)
                blob.update_P1_in_fragment()

def compute_P1_for_blobs_in_video(video, blobs_in_video):
    """Assigns individual-fragment-based identities to all the blobs
    in the video. It uses P1
    """
    for i, frame in enumerate(blobs_in_video):
        for j, blob in enumerate(frame):
            if blob.is_a_fish_in_a_fragment and not blob.assigned_during_accumulation:
                identities_in_fragment = np.asarray(blob.identities_in_fragment())
                frequencies_in_fragment = compute_identification_frequencies_individual_fragment(identities_in_fragment, video.number_of_animals)
                blob._frequencies_in_fragment = frequencies_in_fragment
                blob._P1_vector = compute_P1_individual_fragment_from_blob(frequencies_in_fragment)
                blob.update_P1_in_fragment()


def assign_identity_to_blobs_in_video_by_fragment(video, blobs_in_video):
    """Assigns individual-fragment-based identities to all the blobs
    in the video. It uses P1
    """
    # print('In assign_identity_to_blobs_in_video_by_fragment')
    for blobs_in_frame in blobs_in_video:
        # print("computing P2 in frame")
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment and not blob.previous[0].is_a_fish_in_a_fragment and not blob.assigned_during_accumulation:
                # Get per blob identities in the fragment
                blob._P2_vector = compute_P2_of_individual_fragment_from_blob(blob, blobs_in_video)
                # print("P2", blob._P2_vector)
                # Assign identity to the fragment
                # identity_in_fragment = np.argmax(blob._P1_vector) + 1
                identity_in_fragment = np.argmax(blob._P2_vector) + 1
                # Update identity of all blobs in fragment
                blob.update_identity_in_fragment(identity_in_fragment)

"""functions used during accumulation
but belong to the assign part of the accumulation that's why are here
"""
CERTAINTY_THRESHOLD = 0.5 # threshold to select a individual fragment as eligible for training

def check_certainty_individual_fragment(frequencies_individual_fragment,softmax_probs_median_individual_fragment):
    argsort_frequencies = np.argsort(frequencies_individual_fragment)
    sorted_frequencies = frequencies_individual_fragment[argsort_frequencies]
    sorted_softmax_probs = softmax_probs_median_individual_fragment[argsort_frequencies]
    certainty = np.diff(np.multiply(sorted_frequencies,sorted_softmax_probs)[-2:])/np.sum(sorted_frequencies[-2:])
    # print("sorted_frequencies, ", sorted_frequencies)
    # print("sorted_softmax_probs, ", sorted_softmax_probs)
    # print("certainty of indiv fragment, ", certainty)
    acceptable_individual_fragment = False
    if certainty > CERTAINTY_THRESHOLD:
        acceptable_individual_fragment = True
    else:
        print("global fragment discarded with certainty ", certainty)
    return acceptable_individual_fragment

def assign_identities_to_test_global_fragment(global_fragment, number_of_animals):
    assert global_fragment.used_for_training == False
    global_fragment._temporary_ids = []
    global_fragment._acceptable_for_training = True
    for i, individual_fragment_predictions in enumerate(global_fragment.predictions):
        # compute statistcs
        # print("individual fragment %i" %i)
        identities_in_fragment = np.asarray(individual_fragment_predictions)
        frequencies_in_fragment = compute_identification_frequencies_individual_fragment(identities_in_fragment, number_of_animals)
        # print("frequencies", frequencies_in_fragment)
        P1_of_fragment = compute_P1_individual_fragment_from_blob(frequencies_in_fragment)
        # print("P1", P1_of_fragment)
        # Assign identity to the fragment
        identity_in_fragment = np.argmax(P1_of_fragment)
        global_fragment._temporary_ids.append(identity_in_fragment)
        acceptable_individual_fragment = check_certainty_individual_fragment(P1_of_fragment, global_fragment.softmax_probs_median[i])
        if not acceptable_individual_fragment:
            print("This individual fragment is not good for training")
            global_fragment._acceptable_for_training = False
            break
    print(global_fragment._temporary_ids)
    if not global_fragment.is_unique:
        print("The global fragment is not unique")
        global_fragment._acceptable_for_training = False
    else:
        global_fragment._temporary_ids = np.asarray(global_fragment._temporary_ids)
