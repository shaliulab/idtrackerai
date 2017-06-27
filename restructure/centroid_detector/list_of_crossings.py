from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
import numpy as np
from tqdm import tqdm
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

def get_jump_blobs(blobs_in_video):
    """Collect portraits from blobs if blob.is_a_jump"
    params:
    ------
    blobs_in_video: list of blob objects
        "blobs_in_video" contains the complete collection of blobs detected in
        a video. Each blob object is endowed with relevant attributes (e.g.,
        identity, pixels, frame_number) and methods.
    returns:
    -----
    jumps: list
        list of blobs identified as jumps
    """
    jump_blobs = [] # array of non assigned portrait to be sent to the network for one-shot recognition [to be conditioned wrt 2 * 99perc[velocity]]

    for frame_num, blobs_in_frame in tqdm(enumerate(blobs_in_video), desc = "detect jumps and crossing"):
        print("frame number ", frame_num)
        for blob_num, blob in enumerate(blobs_in_frame):
            print("blob number ", blob_num)
            if blob.identity == 0:
                # if it is a fish, than it has a portrait and can be assigned
                if blob.is_a_fish:
                    #first we assign the exteme points of the individual fragments
                    jump_blobs.append(blob)
    return jump_blobs

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
    def __init__(self, jumping_blob = None, number_of_animals = None, net_prediction = None, softmax_probs = None, velocity_threshold = None):
        self._jumping_blob = jumping_blob
        self.possible_identities = range(1, number_of_animals + 1)
        self.prediction = int(net_prediction)
        self.softmax_probs = softmax_probs
        self.velocity_threshold = velocity_threshold

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


if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector

    #load video and list of blobs
    video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/conflicto_short/session_2/video_object.npy').item()
    number_of_animals = video.number_of_animals
    list_of_blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/conflicto_short/session_2/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    velocity_threshold = compute_model_velocity(blobs, number_of_animals, percentile = VEL_PERCENTILE)
    jump_blobs = get_jump_blobs(blobs)
    jump_images = [blob.portrait[0] for blob in jump_blobs]
    #assign jumps by restoring the network
    assigner = assign_jumps(jump_images, video)

    for i, blob in enumerate(jump_blobs):
        jump = Jump(jumping_blob = blob,
                    number_of_animals = video.number_of_animals,
                    net_prediction = assigner._predictions[i],
                    softmax_probs = assigner._softmax_probs[i],
                    velocity_threshold = velocity_threshold)
        jump.assign_jump(blobs)
        blob._identity = jump.jumping_blob.identity
        blob._P1_vector = assigner._softmax_probs[i]
        blob._P2_vector = None

    frame_by_frame_identity_inspector(video, blobs)
