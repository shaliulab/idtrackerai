from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')

import matplotlib.pyplot as plt
import numpy as np

from network_params import NetworkParams
from get_data import DataSet
from id_CNN import ConvNetwork
from get_predictions import GetPrediction
from blob import get_images_from_blobs_in_video
from visualize_embeddings import EmbeddingVisualiser
from globalfragment import get_images_and_labels_from_global_fragment
from statistics_for_assignment import compute_P1_individual_fragment_from_blob

def assign(video, images, params, print_flag):
    net = ConvNetwork(params, training_flag = False)
    # build data object
    images = np.expand_dims(np.asarray(images), axis = 3)
    data = DataSet(params.number_of_animals, images)
    # Instantiate data_set
    data.standarize_images()
    # Crop images from 36x36 to 32x32 without performing data augmentation
    data.crop_images(image_size = 32)
    # Restore network
    net.restore()
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

def assign_identity_to_blobs_in_video(blobs_in_video, assigner):
    counter = 0
    for frame in blobs_in_video:
        for blob in frame:
            if not blob.is_a_fish_in_a_fragment:
                blob._identity = 0
            elif blob._identity == None:
                blob._identity = int(assigner._predictions[counter])
                counter += 1

def assign_identity_to_blobs_in_fragment(video, blobs_in_video):
    """Assigns individual-fragment-based identities to all the blobs
    in the video. It uses P1
    """
    print('In assign_identity_to_blobs_in_fragment')
    for frame in blobs_in_video:
        for blob in frame:
            if blob.is_a_fish_in_a_fragment and not blob.previous[0].is_a_fish_in_a_fragment:
                # Get per blob identities in the fragment
                identities_in_fragment = np.asarray(blob.identities_in_fragment())
                frequencies_in_fragment = compute_identification_frequencies_individual_fragment(identities_in_fragment, video.number_of_animals)
                P1_of_fragment = compute_P1_individual_fragment_from_blob(frequencies_in_fragment)
                # Assign identity to the fragment
                identity_in_fragment = np.argmax(P1_of_fragment) + 1
                # Update identity of all blobs in fragment
                blob.update_identity_in_fragment(identity_in_fragment)
