from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')
sys.path.append('./postprocessing')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging

from network_params import NetworkParams
from get_data import DataSet
from id_CNN import ConvNetwork
from get_predictions import GetPrediction
from visualize_embeddings import EmbeddingVisualiser
from statistics_for_assignment import compute_P2_of_individual_fragment_from_blob

logger = logging.getLogger("__main__.assigner")

"""
********************************************************************************
assign blobs
********************************************************************************
"""

def assign(net, video, images, print_flag):
    logger.info("assigning identities to images...")
    # build data object
    images = np.expand_dims(np.asarray(images), axis = 3)
    logger.info("generating data set")
    data = DataSet(net.params.number_of_animals, images)
    logger.debug("images shape %s" %str(images.shape))
    data.crop_images(image_size = video.portrait_size[0])
    logger.info("getting predictions")
    assigner = GetPrediction(data, print_flag = print_flag)
    assigner.get_predictions_softmax(net.predict)
    logger.info("done")
    return assigner

"""
********************************************************************************
assign blobs in video
********************************************************************************
"""
def compute_identification_statistics_for_non_accumulated_fragments(fragments, assigner):
    counter = 0

    for fragment in fragments:
        if not fragment.used_for_training and fragment.is_an_individual:
            next_counter_value = counter + fragment.number_of_images
            predictions = assigner.predictions[counter : next_counter_value]
            softmax_probs = assigner.softmax_probs[counter : next_counter_value]
            fragment.compute_identification_statistics(predictions, softmax_probs)
            counter = next_counter_value

def assign_identity(fragments):
    [fragment.assign_identity() for fragment in fragments if fragment.is_an_individual]

"""
********************************************************************************
assign ghost crossings
********************************************************************************
"""
def get_attributes_for_ghost_crossing_assignment(fragment_to_assign, fragments, target_fragment_identifier):
    if target_fragment_identifier is not None:
        attributes = ['identity', 'P1_vector', 'P2_vector']
        for fragment in fragments:
            if fragment.identifier == target_fragment_identifier:
                [setattr(fragment_to_assign, '_' + attribute, getattr(fragment, attribute)) for attribute in attributes
                    if fragment.is_an_individual and hasattr(fragment, 'identity')]

def assign_ghost_crossings(fragments):
    for fragment in tqdm(fragments, desc = "Assigning ghost crossings"):
        if fragment.is_a_ghost_crossing:
            if len(fragment.next_blobs_fragment_identifier) == 1:
                target_fragment_identifier = fragment.next_blobs_fragment_identifier[0]
            elif len(fragment.previous_blobs_fragment_identifier) == 1:
                target_fragment_identifier = fragment.previous_blobs_fragment_identifier[0]
            else:
                target_fragment_identifier = None
            get_attributes_for_ghost_crossing_assignment(fragment, fragments, target_fragment_identifier)


"""
********************************************************************************
main assigner
********************************************************************************
"""
def assigner(list_of_fragments, video, net):
    logger.info("Assigning identities to non-accumulated individual fragments")
    logger.debug("Resetting list of fragments for assignment")
    list_of_fragments.reset(roll_back_to = 'accumulation')
    # Get images from the blob collection
    logger.info("Getting images")
    images = list_of_fragments.get_images_from_fragments_to_assign()
    logger.debug("Images shape before assignment %s" %str(images.shape))
    # get predictions
    logger.info("Getting predictions")
    assigner = assign(net, video, images, print_flag = True)
    logger.debug("Number of generated predictions: %s" %str(len(assigner._predictions)))
    logger.debug("Predictions range: %s" %str(np.unique(assigner._predictions)))
    compute_identification_statistics_for_non_accumulated_fragments(list_of_fragments.fragments, assigner)
    # compute P2 for all the individual fragments (including the already accumulated)
    logger.info("Assigning identities")
    assign_identity(list_of_fragments.fragments)
    logger.info("Assigning ghost crossings")
    assign_ghost_crossings(list_of_fragments.fragments)
