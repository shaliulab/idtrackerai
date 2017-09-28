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
                                    # compute_P1_individual_fragment_from_frequencies,\
                                    # compute_identification_frequencies_individual_fragment,\
                                    # is_assignment_ambiguous

FIXED_IDENTITY_THRESHOLD = .9

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
assign ghost crossings
********************************************************************************
"""

def assign_ghost_crossings(blobs):
    for blobs_in_frame in tqdm(blobs, desc = 'Assign identity to individual fragments extremes'):
        # try:
            # print("\nframe number ", blobs_in_frame[0].frame_number)
        # except:
            # print("last frame")
        for blob in blobs_in_frame:
            #if a blob has not been assigned but it is a fish and overlaps with one fragment
            #assign it!
            if blob.is_a_ghost_crossing:
                print("ghost crossing identity: ", blob.identity)
                print("ghost crossing frame: ", blob.frame_number)
            if (blob.identity == 0 or blob.identity is None) and (blob.is_a_fish or blob.is_a_ghost_crossing):
                print("is a ghost crossing ", blob.is_a_ghost_crossing)
                print("num next ", len(blob.next))
                print("num prev ", len(blob.previous))
                print("identity ", blob.identity)
                if len(blob.next) == 1:
                    print("next is 1")
                    print("next id ", blob.next[0].identity)
                    blob._identity = blob.next[0].identity
                    blob._frequencies_in_fragment = blob.next[0].frequencies_in_fragment
                    blob._P1_vector = blob.next[0].P1_vector
                    blob._P2_vector = blob.next[0].P2_vector ### NOTE: this is not strictly correct as it should be recomputed
                    # blob.is_an_extreme_of_individual_fragment = True
                elif len(blob.previous) == 1:
                    print("prev is 1")
                    print("prev id ", blob.previous[0].identity)
                    blob._identity = blob.previous[0].identity
                    blob._frequencies_in_fragment = blob.previous[0].frequencies_in_fragment
                    blob._P1_vector = blob.previous[0].P1_vector
                    blob._P2_vector = blob.previous[0].P2_vector ### NOTE: this is not strictly correct as it should be recomputed
                    # blob.is_an_extreme_of_individual_fragment = True
                print("assigned during accumulation ----->", blob.assigned_during_accumulation)

"""
********************************************************************************
assign jumps
********************************************************************************
"""

from compute_velocity_model import compute_velocity_from_list_of_blobs, compute_model_velocity

VEL_PERCENTILE = 99 #percentile used to model velocity

class Jump(object):
    def __init__(self, jumping_blob = None, number_of_animals = None, _P2_vector = None, velocity_threshold = None, number_of_frames = None):
        self._jumping_blob = jumping_blob
        self.possible_identities = range(1, number_of_animals + 1)
        self._P2_vector = _P2_vector
        identity_in_fragment = np.argmax(_P2_vector) + 1
        self.prediction = identity_in_fragment
        ambiguous_identity_in_fragment = is_assignment_ambiguous(_P2_vector)
        if ambiguous_identity_in_fragment is list:
            self.prediction = ambiguous_identity_in_fragment
        self.velocity_threshold = velocity_threshold
        self.number_of_frames = number_of_frames
        self.number_of_animals = number_of_animals

    @property
    def jumping_blob(self):
        return self._jumping_blob

    @jumping_blob.setter
    def jumping_blob(self, jumping_blob):
        """by definition, a jumping blob is a blob which satisfied the model area,
        but does not belong to an individual fragment"""
        assert jumping_blob.is_a_fish
        assert not jumping_blob.is_in_a_fragment
        self._jumping_blob = jumping_blob

    def get_available_identities(self, blobs_in_video):
        print('--- getting assigned identities')
        blobs_in_frame_sure_identities = [blob.identity for blob in blobs_in_video[self.jumping_blob.frame_number]
                                            if blob.is_a_fish
                                            and (blob.identity is not None or blob.identity != 0)]
        print("blobs_in_frame_sure_identities: ", blobs_in_frame_sure_identities)
        return set(self.possible_identities) - set(blobs_in_frame_sure_identities)

    def apply_model_velocity(self, blobs_in_video):
        # print("checking velocity model for blob ", self.jumping_blob.identity, " in frame ", self.jumping_blob.frame_number)
        blobs_in_frame = blobs_in_video[self.jumping_blob.frame_number - 1]
        corresponding_blob_list = []
        corresponding_blob_list_past = [blob for blob in blobs_in_frame if blob.is_a_fish and blob.identity == self.jumping_blob.identity]
        if corresponding_blob_list_past:
            corresponding_blob_list.append(corresponding_blob_list_past[0])
        corresponding_blob_list.append(self.jumping_blob)
        # print("self.jumping_blob.frame_number + 1 ", self.jumping_blob.frame_number + 1)
        # print("self.number_of_frames ", self.number_of_frames)
        # print("len(blobs_in_video) ", len(blobs_in_video))
        if self.jumping_blob.frame_number + 1 < self.number_of_frames:
            blobs_in_frame = blobs_in_video[self.jumping_blob.frame_number + 1]
            # for blob in blobs_in_frame:
                # print(blob.frame_number, blob.is_a_fish, blob.identity)
            corresponding_blob_list_future = [blob for blob in blobs_in_frame if blob.is_a_fish and blob.identity == self.jumping_blob.identity]
            # print("corresponding_blob_list_future ", corresponding_blob_list_future)
            if corresponding_blob_list_future:
                corresponding_blob_list.append(corresponding_blob_list_future[0])
            # print("corresponding_blob_list ", corresponding_blob_list)
        if len(corresponding_blob_list) > 1:
            velocity = compute_velocity_from_list_of_blobs(corresponding_blob_list)
            # print("velocity, ", velocity)
            # print("velocity_th, ", self.velocity_threshold)
            return velocity
        else:
            # print("it cannot compute the velocity")
            return 2 * self.velocity_threshold

    def get_prediction_from_P2(self, available_identities):
        print("--- getting P2 vector")
        print("available_identities: ", available_identities)
        print("possible_identities: ", self.possible_identities)
        non_available_identities = np.asarray(list(set(self.possible_identities) - set(available_identities)))
        print("non_available_identities: ", non_available_identities)
        if len(non_available_identities) != 0:
            self._P2_vector[non_available_identities-1] = 0
        prediction = np.where(self._P2_vector == np.max(self._P2_vector))[0] + 1
        if len(prediction) == 1:
            return prediction[0]
        elif len(prediction) > 1:
            return list(prediction)

    def assign_jump(self, blobs_in_video):
        print("\n***** assigning jump")
        print("frame number ", self.jumping_blob.frame_number)
        available_identities = self.get_available_identities(blobs_in_video)
        prediction = self.get_prediction_from_P2(available_identities)
        print("available_identities, ", available_identities)
        print("prediction, ", prediction)
        print("prediction type", type(prediction))
        if type(prediction) is list:
            print("prediction len", len(prediction))
        if type(prediction) is list and len(prediction) > 1 and len(prediction) < self.number_of_animals:
            print("predictions is a list")
            predictions_in_available_identities = [pred for pred in prediction if pred in available_identities]
            if len(predictions_in_available_identities) == 1:
                print("case1")
                # case 1: only one prediction is in the available identities
                prediction = predictions_in_available_identities[0]
            elif len(predictions_in_available_identities) == 0:
                print("case2")
                # case 2: none of the predictions are in the available identities (the prediction has to be in the available identities)
                prediction = prediction[0] # it is solved in the third condition below (in check_assigned_identity)
            elif len(predictions_in_available_identities) > 1:
                print("case3")
                # case 3: more than two predictions are in the available identities (we choose the prediction by the model velocity)
                velocities = []
                for pred in predictions_in_available_identities:
                    self.jumping_blob._identity = pred
                    velocities.append(self.apply_model_velocity(blobs_in_video))
                print("velocities ", velocities)
                velocities = np.asarray(velocities)
                indices_min_velocity = np.where(velocities == np.min(velocities))[0]
                print("min velocity", np.min(velocities))
                if len(indices_min_velocity) == 1:
                    print("there is a unique minimum velocity")
                    prediction = predictions_in_available_identities[indices_min_velocity[0]]
                    print("prediction ", prediction)
                elif len(indices_min_velocity) > 1:
                    print("there are non unique minimum velocity")
                    prediction = 0
                    print("prediction ", prediction)


        if len(available_identities) == 1:
            print("there is only one available identity (%i)" %list(available_identities)[0])
            self.jumping_blob._identity = list(available_identities)[0]
        elif len(available_identities) > 1:
            print("there are more than one available identity")
            self.jumping_blob._identity = prediction
        elif len(available_identities) == 0:
            print("There are no more available identities ---------------------------------------")
            print(self.jumping_blob.frame_number)
            new_identity = -1
        else:
            raise ValueError('condition not considered')

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_frequencies_P1_for_jump(video, blob):
    if not np.any(blob._P1_vector != 0):
        blob._frequencies_in_fragment = np.zeros(video.number_of_animals)
        blob._frequencies_in_fragment[blob.prediction-1] += 1
        if blob.is_a_jumping_fragment and len(blob.next) != 0 and blob.next[0].is_a_jumping_fragment:
            print("next", blob.next)
            print("previous ", blob.previous)
            print("next is jumping fragment", blob.next[0].is_a_jumping_fragment)
            print("cur blob identity", blob.identity)
            print(len(blob.next))
            print(len(blob.next[0].next))
            print(len(blob.next[0].previous))
            print(len(blob.previous))
            # from pprint import pprint
            # pprint(blob.__dict__)
            blob._frequencies_in_fragment[blob.next[0].prediction-1] += 1
            blob._P1_vector = compute_P1_individual_fragment_from_frequencies(blob._frequencies_in_fragment)
            blob.next[0]._frequencies_in_fragment = blob._frequencies_in_fragment
            blob.next[0]._P1_vector = blob._P1_vector
        else: # is a jump or is a identity 0 or None
            blob._P1_vector = compute_P1_individual_fragment_from_frequencies(blob._frequencies_in_fragment)

def compute_P2_for_jump(blob, blobs):
    if not np.any(blob._P2_vector != 0):
        blob._P2_vector = compute_P2_of_individual_fragment_from_blob(blob, blobs)
        if blob.is_a_jumping_fragment and len(blob.next) != 0:
            blob.next[0]._P2_vector = blob._P2_vector

def assign_jumps(images, video):
    """Restore the network associated to the model used to assign video.
    parameters
    ------
    images: ndarray (num_images, height, width)
        "images" collection of images to be assigned
    video: object
        "video" object associated to the tracked video. It contains information
        about the video, the animals tracked, and the status of the tracking.
    return
    -----
    assigner: object
        contains predictions (ndarray of shape [number of images]) of the network, the values in the last fully
        conencted layer (ndarray of shape [number of images, 100]), and the values of the softmax layer
        (ndarray of shape [number of images, number of animals in the tracked video])
    """
    net_params = NetworkParams(video.number_of_animals,
                    learning_rate = 0.005,
                    keep_prob = 1.0,
                    use_adam_optimiser = False,
                    restore_folder = video._accumulation_folder,
                    save_folder = video._accumulation_folder,
                    image_size = video.portrait_size)
    net = ConvNetwork(net_params)
    net.restore()
    return assign(net, video, images, print_flag = True)

def assign_identity_to_jumps(video, blobs):
    if not hasattr(video, "velocity_threshold"):
        video.velocity_threshold = compute_model_velocity(blobs, video.number_of_animals, percentile = VEL_PERCENTILE)
    jump_blobs = [blob for blobs_in_frame in blobs for blob in blobs_in_frame
                    if blob.is_a_jump or (blob.is_a_fish and (blob.identity == 0 or blob.identity is None))]
    # print("number of blobs to assing during jumps, ", len(jump_blobs))
    jump_images = [blob.portrait for blob in jump_blobs]
    #assign jumps by restoring the network
    assigner = assign_jumps(jump_images, video)

    for i, blob in tqdm(enumerate(jump_blobs), desc = 'Assigning predictions to blobs'):
        blob.prediction = int(assigner._predictions[i])

    for blob in tqdm(jump_blobs, desc = 'Computing P1 for jumps'):
        get_frequencies_P1_for_jump(video, blob)

    for blob in tqdm(jump_blobs, desc = 'Computing P2 for jumps'):
        compute_P2_for_jump(blob, blobs)

    # for blob in tqdm(enumerate(jump_blobs), desc = 'Assigning identity to jumps'):
    original_len_jump_blobs = len(jump_blobs)
    while len(jump_blobs) > 0:
        # print(len(jump_blobs), '/', original_len_jump_blobs)
        blob = jump_blobs[get_blob_to_assign_by_max_P2(jump_blobs)]
        # print("\n\nframe number, ", blob.frame_number)
        # print("fragment identifier, ", blob.fragment_identifier)
        # print("blob identity before assigning jump ", blob.identity)
        jump = Jump(jumping_blob = blob,
                    number_of_animals = video.number_of_animals,
                    _P2_vector = blob._P2_vector,
                    velocity_threshold = video.velocity_threshold,
                    number_of_frames = video.number_of_frames)

        jump.assign_jump(blobs)
        blob._identity = jump.jumping_blob.identity
        if len(blob.next) == 1 and blob.next[0].is_a_fish and blob.next[0].fragment_identifier == blob.fragment_identifier:
            print("assigning identity %i to next in frame %i " %(blob.identity, blob.next[0].frame_number))
            blob.next[0]._identity = blob.identity
            if blob.next[0] in jump_blobs: jump_blobs.remove(blob.next[0])
        jump_blobs.remove(blob)

"""
********************************************************************************
assign blobs in video
********************************************************************************
"""

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

                    while len(current.next) > 0 and current.next[0].fragment_identifier == blob.fragment_identifier:
                        current = current.next[0]
                        current._identity = int(assigner._predictions[counter])
                        counter += 1

                    current = blob

                    while len(current.previous) > 0 and current.previous[0].fragment_identifier == blob.fragment_identifier:
                        current = current.previous[0]
                        current._identity = int(assigner._predictions[counter])
                        counter += 1

def assign_identity_to_blobs_in_video_by_fragment(video, blobs_in_video):
    """Assigns individual-fragment-based identities to all the blobs
    in the video.
    """
    assigned_fragment_identifiers = []
    list_of_blobs = get_blobs_to_assign(blobs_in_video, assigned_fragment_identifiers)
    len_first_list_of_blobs = len(list_of_blobs)

    while len(list_of_blobs) > 1:
        blob = list_of_blobs[get_blob_to_assign_by_max_P2(list_of_blobs)]

        logger.info("frame number: %i" %blob.frame_number)
        logger.info("certainty of the assignment (P2): %s" %str(max(blob.P2_vector)))
        identity_in_fragment = np.argmax(blob._P2_vector) + 1
        ambiguous_identities, is_ambiguous_identity = is_assignment_ambiguous(blob.P2_vector)
        if is_ambiguous_identity:
            logger.debug("******frame number: %i" %blob.frame_number)
            logger.debug("assigned_during_accumulation: %s" %blob.assigned_during_accumulation)
            logger.debug("identity_in_fragment (ambiguous): %s" %str(ambiguous_identities))
            identity_in_fragment = 0
            blob.ambiguous_identities = ambiguous_identities
        logger.debug("identity_in_fragment: %i" %identity_in_fragment)
        # Update identity of all blobs in fragment
        number_of_images_in_fragment = len(blob.identities_in_fragment())
        logger.debug("number_of_images_in_fragment: %i" %number_of_images_in_fragment)
        blob.update_identity_in_fragment(identity_in_fragment, number_of_images_in_fragment = number_of_images_in_fragment)
        # blob.update_attributes_in_fragment(['_identity'], [identity_in_fragment])
        assigned_fragment_identifiers.append(blob.fragment_identifier)
        # list_of_blobs = get_blobs_to_assign(blobs_in_video, assigned_fragment_identifiers)
        list_of_blobs.remove(blob)
        assert blob not in list_of_blobs
        logger.info("step %i of %i" %(len(list_of_blobs), len_first_list_of_blobs))
        coexisting_blobs, _ = blob.get_coexisting_blobs_in_fragment(blobs_in_video)
        if np.max(blob._P2_vector) < FIXED_IDENTITY_THRESHOLD:
            for blob_to_assign in tqdm(coexisting_blobs, desc = "Updating P2 of coexisting blobs"):
                if np.max(blob_to_assign._P2_vector) < FIXED_IDENTITY_THRESHOLD:
                    blob_to_assign._P2_vector = compute_P2_of_individual_fragment_from_blob(blob_to_assign, blobs_in_video)
                    blob_to_assign.update_attributes_in_fragment(['_P2_vector'], [blob_to_assign._P2_vector])

"""
********************************************************************************
P1 and P2
********************************************************************************
"""

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
    individual_fragments_identifiers_computed = []

    for blobs_in_frame in tqdm(blobs_in_video, desc = 'Computing P2 vectors'):
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment\
                and blob._fragment_identifier not in individual_fragments_identifiers_computed:
                # Get per blob identities in the fragment
                blob._P2_vector = compute_P2_of_individual_fragment_from_blob(blob, blobs_in_video)
                # update P2 for the blobs part of the current individual fragment
                blob.update_attributes_in_fragment(['_P2_vector'], [blob._P2_vector])
                individual_fragments_identifiers_computed.append(blob._fragment_identifier)

"""
********************************************************************************
get blobs to assign
********************************************************************************
"""

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

def get_blob_to_assign_by_max_P2(list_of_blobs):
    return np.argmax(np.asarray([max(blob.P2_vector) for blob in list_of_blobs]))

"""
********************************************************************************
main assigner
********************************************************************************
"""

def assigner(blobs, video, net):

    logger.info("Assigning identities to non-accumulated individual fragments")
    logger.info("Preparing blob objects")
    reset_blobs_fragmentation_parameters(blobs, recovering_from = 'assignment')
    # Get images from the blob collection
    logger.info("Getting images")
    images = get_images_from_blobs_in_video(blobs)#, video._episodes_start_end)
    logger.debug("Images shape before assignment %s" %str(images.shape))
    # get predictions
    logger.info("Getting predictions")
    assigner = assign(net, video, images, print_flag = True)
    logger.debug("Number of generated predictions: %s" %str(len(assigner._predictions)))
    logger.debug("Predictions range: %s" %str(np.unique(assigner._predictions)))
    # assign identities to each blob in each frame
    logger.info("Assigning identities to individual fragments")
    assign_identity_to_blobs_in_video(blobs, assigner)
    # compute P1 vector for individual fragmets
    logger.debug("Computing P1")
    compute_P1_for_blobs_in_video(video, blobs)
    # compute P2 for all the individual fragments (including the already accumulated)
    logger.debug("Computing P2")
    compute_P2_for_blobs_in_video(video, blobs)
    # assign identities based on individual fragments
    logger.debug("Assigning identities on an individual fragment basis")
    assign_identity_to_blobs_in_video_by_fragment(video, blobs)
    # assign identity to ghost crossings
    logger.debug("Assigning identities to ghost crossings")
    assign_ghost_crossings(blobs)
    # solve jumps
    logger.debug("Assigning identities to jumps")
    assign_identity_to_jumps(video, blobs)
