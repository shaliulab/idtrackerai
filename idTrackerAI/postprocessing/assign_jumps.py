from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import collections


from blob import ListOfBlobs
from assigner import assign, get_blob_to_assign
from statistics_for_assignment import compute_P2_of_individual_fragment_from_blob, is_assignment_ambiguous, compute_P1_individual_fragment_from_frequencies
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
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

    # def check_id_availability(self, available_identities, sorted_assignments_indices):
    #     return [sorted_assignments_index + 1  for sorted_assignments_index in sorted_assignments_indices
    #         if (sorted_assignments_index + 1) in available_identities]

    # def check_assigned_identity(self, blobs_in_video, available_identities, sorted_assignments_indices):
    #     if not self.apply_model_velocity(blobs_in_video):
    #         # print("\navailable_identities ", available_identities)
    #         # print("removing ", self.jumping_blob.identity)
    #         if len(list(available_identities)) > 0:
    #             available_identities.remove(self.jumping_blob.identity)
    #             # print("new_available_identities ", available_identities)
    #         if len(list(available_identities)) > 0:
    #             self.jumping_blob.identity = self.check_id_availability(available_identities, sorted_assignments_indices)[0]
    #             # print("self.check_id_availability(available_identities, sorted_assignments_indices), ", self.check_id_availability(available_identities, sorted_assignments_indices))
    #             # print("self.jumping_blob.identity, ", self.jumping_blob.identity)
    #             self.check_assigned_identity(blobs_in_video, available_identities, sorted_assignments_indices)
    #         else:
    #             # print("no more available_identities")
    #             self.jumping_blob.identity = 0
        # else:
        #     print("it passes the velocity model")
        #     print("self.jumping_blob.identity, ", self.jumping_blob.identity)

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

        # print("prediciton", self.prediction)
        # if self.jumping_blob.frame_number >= 1 and self.jumping_blob.identity != 0:
        #     print("checking the velocity model")
        #     sorted_assignments_indices = np.argsort(np.array(self._P2_vector))[::-1]
        #     self.check_assigned_identity(blobs_in_video, available_identities, sorted_assignments_indices)

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
        if blob.is_a_jumping_fragment and len(blob.next) != 0:
            # print("next", blob.next)
            # print("previous ", blob.previous)
            # print("next is jumping fragment", blob.next[0].is_a_jumping_fragment)
            # print("cur blob identity", blob.identity)
            # print(len(blob.next))
            # print(len(blob.next[0].next))
            # print(len(blob.next[0].previous))
            # print(len(blob.previous))
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
        blob = jump_blobs[get_blob_to_assign(jump_blobs)]
        # print("\n\nframe number, ", blob.frame_number)
        # print("fragment identifier, ", blob.fragment_identifier)
        # print("blob identity before assigning jump ", blob.identity)
        jump = Jump(jumping_blob = blob,
                    number_of_animals = video.number_of_animals,
                    _P2_vector = blob._P2_vector,
                    velocity_threshold = video.velocity_threshold,
                    number_of_frames = video._num_frames)

        jump.assign_jump(blobs)
        blob._identity = jump.jumping_blob.identity
        if len(blob.next) == 1 and blob.next[0].is_a_fish and blob.next[0].fragment_identifier == blob.fragment_identifier:
            print("assigning identity %i to next in frame %i " %(blob.identity, blob.next[0].frame_number))
            blob.next[0]._identity = blob.identity
            if blob.next[0] in jump_blobs: jump_blobs.remove(blob.next[0])
        jump_blobs.remove(blob)
