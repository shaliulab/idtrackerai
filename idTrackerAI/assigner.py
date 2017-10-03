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
from blob import get_images_from_blobs_in_video, reset_blobs_fragmentation_parameters
from visualize_embeddings import EmbeddingVisualiser
from list_of_global_fragments import get_images_and_labels_from_global_fragment
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
        if not fragment.used_for_training and fragment.is_a_fish:
            next_counter_value = counter + fragment.number_of_images
            predictions = assigner.predictions[counter : next_counter_value]
            softmax_probs = assigner.softmax_probs[counter : next_counter_value]
            fragment.compute_identification_statistics(predictions, softmax_probs)
            counter = next_counter_value

def assign_identity(fragments):
    [fragment.assign_identity() for fragment in fragments if fragment.is_a_fish]

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
