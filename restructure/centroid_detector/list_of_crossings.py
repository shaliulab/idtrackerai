from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
import numpy as np
from tqdm import tqdm
import collections
from blob import ListOfBlobs
import matplotlib.pyplot as plt
from py_utils import get_spaced_colors_util
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob

"""if the blob was not assigned durign the standard procedure it could be a jump or a crossing
these blobs are blobs that are not in a fragment. We distinguish the two cases in which
they are single fish (blob.is_a_fish == True) or crossings (blob.is_a_fish == False)
"""
VEL_PERCENTILE = 99

def compute_model_velocity(blobs_in_video, number_of_animals, percentile = VEL_PERCENTILE):
    """computes the 2 * (99 percentile) of the distribution of velocities of identified fish.
    params
    -----
    blobs_in_video: list of blob objects
        collection of blobs detected in the video.
    number_of_animals int
    percentile int

    return
    -----
    float
    2* percentile(velocity distribution of identified animals)
    """
    distance_travelled_in_individual_fragments = []
    current_individual_fragment_identifier = -1

    for blobs_in_frame in tqdm( blobs_in_video, desc = "computing velocity model"):

        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment and current_individual_fragment_identifier != blob.fragment_identifier:
                current_individual_fragment_identifier = blob.fragment_identifier
                distance_travelled_in_individual_fragments.extend(blob.frame_by_frame_velocity())

    # return 2 * np.percentile(distance_travelled_in_individual_fragments, percentile)
    return 2 * np.max(distance_travelled_in_individual_fragments)

def assign_jump_portraits(images, video):
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
    assert video._has_been_assigned == True
    net_params = NetworkParams(video.number_of_animals,
                    learning_rate = 0.005,
                    keep_prob = 1.0,
                    use_adam_optimiser = False,
                    restore_folder = video._accumulation_folder,
                    save_folder = video._accumulation_folder)
    net = ConvNetwork(net_params)
    net.restore()
    return assign(net, video, images, print_flag = True)

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
    assert video._has_been_assigned == True
    net_params = NetworkParams(video.number_of_animals,
                    learning_rate = 0.005,
                    keep_prob = 1.0,
                    use_adam_optimiser = False,
                    restore_folder = video._accumulation_folder,
                    save_folder = video._accumulation_folder)
    net = ConvNetwork(net_params)
    net.restore()
    return assign(net, video, images, print_flag = True)

def compute_velocity_from_list_of_blobs(list_of_blobs):
    centroids = [blob.centroid for blob in list_of_blobs]
    print(centroids)
    velocity = [np.linalg.norm(centroids[i+1] - centroid) for i, centroid in enumerate(centroids[:-1])]
    print(velocity)
    return np.mean(velocity)


class Jump(object):
    def __init__(self, jumping_blob = None, number_of_animals = None, net_prediction = None, softmax_probs = None, velocity_threshold = None, number_of_frames = None):
        self._jumping_blob = jumping_blob
        self.possible_identities = range(1, number_of_animals + 1)
        self.prediction = int(net_prediction)
        self.softmax_probs = softmax_probs
        self.velocity_threshold = velocity_threshold
        self.number_of_frames = number_of_frames

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
        blobs_in_frame_sure_identities = [blob.identity for blob in blobs_in_video[self.jumping_blob.frame_number] if blob.identity != 0]
        return set(self.possible_identities) - set(blobs_in_frame_sure_identities)

    def apply_model_velocity(self, blobs_in_video):
        print("checking velocity model for blob ", self.jumping_blob.identity, " in frame ", self.jumping_blob.frame_number)
        blobs_in_frame = blobs_in_video[self.jumping_blob.frame_number - 1]
        corresponding_blob_list = []
        corresponding_blob_list_past = [blob for blob in blobs_in_frame if blob.is_a_fish and blob.identity == self.jumping_blob.identity]
        if corresponding_blob_list_past:
            corresponding_blob_list.append(corresponding_blob_list_past[0])
        corresponding_blob_list.append(self.jumping_blob)
        print("self.jumping_blob.frame_number + 1 ", self.jumping_blob.frame_number + 1)
        print("self.number_of_frames ", self.number_of_frames)
        print("len(blobs_in_video) ", len(blobs_in_video))
        if self.jumping_blob.frame_number + 1 < self.number_of_frames:
            blobs_in_frame = blobs_in_video[self.jumping_blob.frame_number + 1]
            corresponding_blob_list_future = [blob for blob in blobs_in_frame if blob.is_a_fish and blob.identity == self.jumping_blob.identity]
            if corresponding_blob_list_future:
                corresponding_blob_list.append(corresponding_blob_list_future[0])
            print("corresponding_blob_list ", corresponding_blob_list)
        if len(corresponding_blob_list) > 1:
            velocity = compute_velocity_from_list_of_blobs(corresponding_blob_list)
            print("velocity, ", velocity)
            print("velocity_th, ", self.velocity_threshold)
            return velocity < self.velocity_threshold
        else:
            return False

    def check_id_availability(self, available_identities, sorted_assignments_indices):
        return [sorted_assignments_index + 1  for sorted_assignments_index in sorted_assignments_indices
            if (sorted_assignments_index + 1) in available_identities]

    def check_assigned_identity(self, blobs_in_video, available_identities, sorted_assignments_indices):
        if not self.apply_model_velocity(blobs_in_video):
            print("available_identities ", available_identities)
            print("removing ", self.jumping_blob.identity)
            available_identities.remove(self.jumping_blob.identity)
            print("new_available_identities ", available_identities)
            if len(list(available_identities)) > 0:
                self.jumping_blob.identity = self.check_id_availability(available_identities, sorted_assignments_indices)[0]
                print("self.check_id_availability(available_identities, sorted_assignments_indices), ", self.check_id_availability(available_identities, sorted_assignments_indices))
                print("self.jumping_blob.identity, ", self.jumping_blob.identity)
                self.check_assigned_identity(blobs_in_video, available_identities, sorted_assignments_indices)
            else:
                print("no more available_identities")
                self.jumping_blob.identity = 0
        else:
            print("it passes the velocity model")
            print("self.jumping_blob.identity, ", self.jumping_blob.identity)


    def assign_jump(self, blobs_in_video):
        available_identities = self.get_available_identities(blobs_in_video)

        if self.prediction in available_identities:
            self.jumping_blob._identity = self.prediction
        else:
            sorted_assignments_indices = np.argsort(np.array(self.softmax_probs))[::-1]
            new_identity = [sorted_assignments_index for sorted_assignments_index in sorted_assignments_indices
                if (sorted_assignments_index + 1) in available_identities][0]
            self.jumping_blob._identity = new_identity + 1

        if self.jumping_blob.frame_number >= 1:
            sorted_assignments_indices = np.argsort(np.array(self.softmax_probs))[::-1]
            self.check_assigned_identity(blobs_in_video, available_identities, sorted_assignments_indices)

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

class Duplication(object):
    def __init__(self, blobs_in_frame_with_duplication = [], duplicated_identities = []):
        ''' Solve duplications and impossible shits of identity (according to velocity_threshold) '''
        self.blobs_in_frame = blobs_in_frame_with_duplication
        self.blobs_assigned_during_accumulation = [blob for blob in blobs_in_frame if blob.assigned_during_accumulation]
        self.identities_assigned_during_accumulation = np.asarray([blob.identity for blob in self.blobs_assigned_during_accumulation])
        self.possible_identities = range(1, self.blobs_in_frame[0].number_of_animals + 1)

    def get_blobs_to_be_reassigned(self, duplicated_identity):
        self.blob_to_be_reassigned = [blob for blob in self.blobs_in_frame
                    if blob.identity == duplicated_identity
                    and blob not in self.blobs_assigned_during_accumulation]

    def get_available_identities(self):
        return set(self.possible_identities) - set(self.identities_assigned_during_accumulation)

    def reassign_identities(self):
        
        for duplicated_identity in self.duplicated_identities:
            self.reassign_identity(duplicated_identity)

    @staticmethod
    def get_P2_vectors(blobs):
        return np.asmatrix([blob.P2_vector for blob in blobs])

    def set_to_zero_P2_values_of_protected_ids(self, P2_matrix):
        #get indices from identities
        indices_to_zero = self.identities_assigned_during_accumulation - 1
        #put the corresponding rows to zero
        P2_matrix = np.delete(P2_matrix, indices_to_zero, 0)                           
        #ditto for the columns
        P2_matrix[:, indices_to_zero] = 0
        return P2_matrix

    def give_unique_identities(self, P2_matrix):
        P2_argsort = np.flip(np.argsort(P2_matrix, axis = 1), axis = 1)
        corresponding_P2s = np.flip(np.sort(P2_matrix, axis = 1), axis = 1)
        #sweep through the columns of P2_argsort to assign the identities


    def reassign_identity(self, duplicated_identity):
        """Get information about the animals contained in a frame in which
        a duplication occured
        """
        blobs_to_reassign = self.get_blobs_to_be_reassigned(duplicated_identity)
        P2_matrix = self.get_P2_vectors(blobs_to_reassign)
        P2_matrix = self.set_to_zero_P2_values_of_protected_ids(P2_matrix,
                                    self.blobs_assigned_during_accumulation)
        identities_to_reassign = give_unique_identities(P2_matrix)


if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector
    NUM_CHUNKS_BLOB_SAVING = 10

    #load video and list of blobs
    # video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/video_object.npy').item()
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/video_object.npy').item()
    number_of_animals = video.number_of_animals
    # list_of_blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/preprocessing/blobs_collection.npy'
    list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/preprocessing/blobs_collection_safe.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    if not hasattr(video, "velocity_threshold"):
        video.velocity_threshold = compute_model_velocity(blobs, number_of_animals, percentile = VEL_PERCENTILE)

    for blobs_in_frame in blobs:
        print("frame_number: ", blobs_in_frame[0].frame_number)
        identities = [blob.identity for blob in blobs_in_frame if blob.identity != 0]
        duplicated_identities = set([x for x in identities if identities.count(x) > 1])
        if len(duplicated_identities) > 0:
            frame  = Duplication(blobs_in_frame_with_duplication = blobs_in_frame,
                                duplicated_identities = duplicated_identities)
            frame.reassign_identities()

    ''' Assign identities to jumps '''
    jump_blobs = [blob for blobs_in_frame in blobs for blob in blobs_in_frame
                    if blob.is_a_jump or blob.is_a_ghost_crossing]
    jump_images = [blob.portrait[0] for blob in jump_blobs]
    #assign jumps by restoring the network
    assigner = assign_jumps(jump_images, video)

    for i, blob in enumerate(jump_blobs):
        jump = Jump(jumping_blob = blob,
                    number_of_animals = video.number_of_animals,
                    net_prediction = assigner._predictions[i],
                    softmax_probs = assigner._softmax_probs[i],
                    velocity_threshold = video.velocity_threshold,
                    number_of_frames = video._num_frames)
        jump.assign_jump(blobs)
        blob._identity = jump.jumping_blob.identity
        blob._P1_vector = assigner._softmax_probs[i]
        blob._P2_vector = None

    ''' Find who is in the crossings '''





    # crossing_identifier = 0
    #
    # for frame_number, blobs_in_frame in enumerate(blobs):
    #     ''' from past to future '''
    #     print("---------------frame_number (from past): ", frame_number)
    #     for blob in blobs_in_frame:
    #         print('***new blob ')
    #         if blob.is_a_crossing:
    #             print('this blob is a crossing')
    #             blob._identity = list(flatten([previous_blob.identity for previous_blob in blob.previous]))
    #             blob.bad_crossing = False
    #             for previous_blob in blob.previous:
    #                 print("\nprevious_blob: is_a_fish - %i, is_a_crossing - %i" %(previous_blob.is_a_fish, previous_blob.is_a_crossing))
    #                 print("previous_blob identity: ", previous_blob.identity)
    #                 if previous_blob.is_a_crossing:
    #                     print("previous_blob_next_crossings ", [previous_blob_next.is_a_crossing for previous_blob_next in previous_blob.next])
    #                     for previous_blob_next in previous_blob.next:
    #                         print('--->', previous_blob_next.identity)
    #                     previous_has_more_than_one_crossing = sum([previous_blob_next.is_a_crossing for previous_blob_next in previous_blob.next]) > 1
    #                     print("previous_has_more_than_one_crossing, ", previous_has_more_than_one_crossing)
    #                     if previous_has_more_than_one_crossing:
    #                         blob.bad_crossing = True
    #                     if len(previous_blob.next) != 1: # the previous crossing_blob is splitting
    #                         for previous_blob_next in previous_blob.next:
    #                             print("previous_blob_next: is_a_fish - %i, is_a_crossing - %i" %(previous_blob_next.is_a_fish, previous_blob_next.is_a_crossing))
    #                             print("previous_blob_next identity: ", previous_blob_next.identity)
    #                             if previous_blob_next is not blob: # for every next of the previous that is not the current blob we remove the identities
    #                                 print(previous_blob_next.identity)
    #                                 if previous_blob_next.is_a_fish and previous_blob_next.identity != 0 and previous_blob_next.identity in blob._identity:
    #                                     blob._identity.remove(previous_blob_next.identity)
    #                                 else:
    #                                     print('we do nothing, probably a badly solved jump')
    #
    #         blob.crossing_identifier = crossing_identifier
    #         crossing_identifier += 1
    #
    #         print("blob.identity: ", blob.identity)

    # for frame_number, blobs_in_frame in enumerate(blobs[::-1]):
    #     ''' from future to past '''
    #     for blob in blobs_in_frame:
    #         print("\nframe_number (from future): ", blob.frame_number)
    #         if blob.is_a_crossing:
    #             print(blob.is_a_crossing)
    #             has_more_than_one_crossing = sum([blob_previous.is_a_crossing for blob_previous in blob.previous]) > 1
    #             print("has_more_than_one_crossing, ", has_more_than_one_crossing)
    #             for blob_previous in blob.previous:
    #                 if blob_previous.is_a_crossing:
    #                     print("previous_blob.bad_crossing (before), ", blob_previous.bad_crossing)
    #                 if blob_previous.is_a_crossing and blob_previous.bad_crossing and has_more_than_one_crossing:
    #                     blob_previous.bad_crossing = True
    #                     print("previous_blob.bad_crossing(after), ", blob_previous.bad_crossing)
    #             blob._identity.extend(list(flatten([next_blob.identity for next_blob in blob.next])))
    #             blob._identity = list(np.unique(blob._identity))
    #             for next_blob in blob.next:
    #                 print("next_blob: is_a_fish - %i, is_a_crossing - %i" %(next_blob.is_a_fish, next_blob.is_a_crossing))
    #                 print("next_blob identity: ", next_blob.identity)
    #                 if next_blob.is_a_crossing:
    #                     if len(next_blob.previous) != 1: # the next crossing_blob is splitting
    #                         for next_blob_previous in next_blob.previous:
    #                             print("next_blob_previous: is_a_fish - %i, is_a_crossing - %i" %(next_blob_previous.is_a_fish, next_blob_previous.is_a_crossing))
    #                             if next_blob_previous is not blob:
    #                                 print(next_blob_previous.identity)
    #                                 if next_blob_previous.is_a_fish and next_blob_previous.identity != 0 and next_blob_previous.identity in blob._identity:
    #                                     blob._identity.remove(next_blob_previous.identity)
    #                                 elif next_blob_previous.is_a_crossing and not next_blob_previous.bad_crossing:
    #                                     [blob._identity.remove(identity) for identity in next_blob_previous.identity if identity in blob._identity]
    #                                 else:
    #                                     print('we do nothing, probably a badly solved jump')
    #
    #             identities_to_remove_from_crossing = [blob_to_remove.identity for blob_to_remove in blobs_in_frame if blob_to_remove.is_a_fish]
    #             identities_to_remove_from_crossing.extend([0])
    #             [blob._identity.remove(identity) for identity in identities_to_remove_from_crossing if identity in blob._identity]
    #             if blob.bad_crossing:
    #                 blob.number_of_animals_in_crossing = None
    #             else:
    #                 blob.number_of_animals_in_crossing = len(blob.identity)
    #         print("blob.identity: ", blob.identity)

    # frame_by_frame_identity_inspector(video, blobs)
    blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
    blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
    blobs_list.cut_in_chunks()
    blobs_list.save()
