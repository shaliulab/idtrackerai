from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from network_params import NetworkParams
from get_data import DataSet
from id_CNN import ConvNetwork
from get_predictions import GetPrediction
from blob import get_images_from_blobs_in_video
from visualize_embeddings import EmbeddingVisualiser
from globalfragment import get_images_and_labels_from_global_fragment
from statistics_for_assignment import compute_P2_of_individual_fragment_from_blob,\
                                    compute_P1_individual_fragment_from_frequencies,\
                                    compute_identification_frequencies_individual_fragment,\
                                    is_assignment_ambiguous

def assign(net, video, images, print_flag):
    print("assigning identities to images...")
    # build data object
    images = np.expand_dims(np.asarray(images), axis = 3)
    data = DataSet(net.params.number_of_animals, images)
    # Instantiate data_set
    # print("standarizing...")
    # data.standarize_images()
    # Crop images from 36x36 to 32x32 without performing data augmentation
    # print("cropping...")
    print("images shape ", images.shape)
    data.crop_images(image_size = video.portrait_size[0])
    # Train network
    # print("getting predictions...")
    assigner = GetPrediction(data, print_flag = print_flag)
    assigner.get_predictions_softmax(net.predict)
    return assigner

# def assign_identity_to_blobs_in_video(blobs_in_video, assigner):
#     counter = 0
#     for blobs_in_frame in blobs_in_video:
#         for blob in blobs_in_frame:
#             if blob.is_a_fish_in_a_fragment and not blob.assigned_during_accumulation:
#                 assert blob._identity == None
#                 blob._identity = int(assigner._predictions[counter])
#                 counter += 1
#             elif not blob.is_a_fish_in_a_fragment:
#                 blob._identity = 0

def assign_identity_to_blobs_in_video(blobs_in_video, assigner):
    counter = 0
    fragments_identifier_used = []
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if not blob.assigned_during_accumulation and blob._fragment_identifier not in fragments_identifier_used:

                if blob.is_a_fish_in_a_fragment:
                    fragments_identifier_used.append(blob.fragment_identifier)
                    current = blob
                    current._identity = int(assigner._predictions[counter])
                    counter += 1
                    while current.next[0].is_a_fish_in_a_fragment:
                        current = current.next[0]
                        current._identity = int(assigner._predictions[counter])
                        counter += 1
                    if len(current.next) == 1 and len(current.next[0].previous) == 1 and current.next[0].is_a_fish:
                        current.next[0]._identity = int(assigner._predictions[counter])
                        counter += 1
                    # elif current.next[0].is_a_ghost_crossing:
                    #     current.next[0]._identity = int(assigner._predictions[counter])
                    #     counter += 1

                    current = blob
                    while current.previous[0].is_a_fish_in_a_fragment:
                        current = current.previous[0]
                        current._identity = int(assigner._predictions[counter])
                        counter += 1
                    if len(current.previous) == 1 and len(current.previous[0].next) == 1 and current.previous[0].is_a_fish:
                        current.previous[0]._identity = int(assigner._predictions[counter])
                        counter += 1
                    # elif current.previous[0].is_a_ghost_crossing:
                    #     current.previous[0]._identity = int(assigner._predictions[counter])
                    #     counter += 1

def compute_P1_for_blobs_in_video(video, blobs_in_video):
    """Assigns individual-fragment-based identities to all the blobs
    in the video. It uses P1
    """
    individual_fragments_identifiers_computed = []
    for blobs_in_frame in tqdm(blobs_in_video, desc = 'Computing P1 vectors'):
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment\
                and not blob.assigned_during_accumulation\
                and blob._fragment_identifier not in individual_fragments_identifiers_computed:

                identities_in_fragment = np.asarray(blob.identities_in_fragment())
                non_shared_information_in_fragment = np.asarray(blob.non_shared_information_in_fragment())
                frequencies_in_fragment = compute_identification_frequencies_individual_fragment(non_shared_information_in_fragment,
                                                                                                    identities_in_fragment,
                                                                                                    video.number_of_animals)
                blob._frequencies_in_fragment = frequencies_in_fragment
                blob._P1_vector = compute_P1_individual_fragment_from_frequencies(frequencies_in_fragment)
                blob.update_attributes_in_fragment(['_P1_vector', '_frequencies_in_fragment'], [blob._P1_vector, blob._frequencies_in_fragment])
                individual_fragments_identifiers_computed.append(blob._fragment_identifier)

def compute_P2_for_blobs_in_video(video, blobs_in_video):
    """compute P2 for all the blobs in the video.
    """
    # print('In assign_identity_to_blobs_in_video_by_fragment')
    individual_fragments_identifiers_computed = []
    for blobs_in_frame in tqdm(blobs_in_video, desc = 'Computing P2 vectors'):
        # print("computing P2 in frame")
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment\
                and blob._fragment_identifier not in individual_fragments_identifiers_computed:
                # Get per blob identities in the fragment
                blob._P2_vector = compute_P2_of_individual_fragment_from_blob(blob, blobs_in_video)
                blob.update_attributes_in_fragment(['_P2_vector'], [blob._P2_vector])
                individual_fragments_identifiers_computed.append(blob._fragment_identifier)

def get_blobs_to_assign(blobs_in_video, assigned_fragment_identifiers):
    blobs_to_assign = []
    used_fragment_identifiers = []

    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment \
                and blob.fragment_identifier not in assigned_fragment_identifiers\
                and blob.fragment_identifier not in used_fragment_identifiers\
                and not blob.assigned_during_accumulation:

                blobs_to_assign.append(blob)
                used_fragment_identifiers.append(blob.fragment_identifier)

    return blobs_to_assign

def get_blob_to_assign(list_of_blobs):
    return np.argmax(np.asarray([max(blob.P2_vector) for blob in list_of_blobs]))


def assign_identity_to_blobs_in_video_by_fragment(video, blobs_in_video):
    """Assigns individual-fragment-based identities to all the blobs
    in the video.
    """
    assigned_fragment_identifiers = []
    list_of_blobs = get_blobs_to_assign(blobs_in_video, assigned_fragment_identifiers)
    len_first_list_of_blobs = len(list_of_blobs)

    while len(list_of_blobs) > 1:
        blob = list_of_blobs[get_blob_to_assign(list_of_blobs)]
        print("\nframe number ", blob.frame_number)
        print("P2 max for blob to be assigned ", max(blob.P2_vector))
        identity_in_fragment = np.argmax(blob._P2_vector) + 1
        ambiguous_identities, is_ambiguous_identity = is_assignment_ambiguous(blob.P2_vector)
        if is_ambiguous_identity:
            print("******frame", blob.frame_number)
            print("assigned_during_accumulation ", blob.assigned_during_accumulation)
            print("identity_in_fragment (ambiguous) ", ambiguous_identities)
            identity_in_fragment = 0
            blob.ambiguous_identities = ambiguous_identities
        print("identity_in_fragment, ", identity_in_fragment)
        # Update identity of all blobs in fragment
        number_of_images_in_fragment = len(blob.identities_in_fragment())
        print("number_of_images_in_fragment, ", number_of_images_in_fragment)
        blob.update_identity_in_fragment(identity_in_fragment, number_of_images_in_fragment = number_of_images_in_fragment)
        # blob.update_attributes_in_fragment(['_identity'], [identity_in_fragment])
        assigned_fragment_identifiers.append(blob.fragment_identifier)
        list_of_blobs = get_blobs_to_assign(blobs_in_video, assigned_fragment_identifiers)
        assert blob not in list_of_blobs
        print("fragments to go ", len(list_of_blobs), " / ", len_first_list_of_blobs)


        for blob_to_assign in list_of_blobs:
            blob._P2_vector = compute_P2_of_individual_fragment_from_blob(blob, blobs_in_video)
            blob.update_attributes_in_fragment(['_P2_vector'], [blob._P2_vector])
