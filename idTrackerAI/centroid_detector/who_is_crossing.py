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

"""if the blob was not assigned durign the standard procedure it could be a jump or a crossing
these blobs are blobs that are not in a fragment. We distinguish the two cases in which
they are single fish (blob.is_a_fish == True) or crossings (blob.is_a_fish == False)
"""
VEL_PERCENTILE = 99

def get_jumps_and_crossing_frames_arrays(blobs_in_video):
    """Given a collection of blobs finds the ones without an assigned identity
    and splits them in two classes: the blob is saved in the list "jumps" if
    it has been classified as being a fish. Otherwise, the frame number is added
    to the list "crossing_frames"
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
    crossing_frames: list
        list of frames indices in which a crossing occurs
    """
    crossing_frames = []
    jumps = [] # array of non assigned portrait to be sent to the network for one-shot recognition [to be conditioned wrt 2 * 99perc[velocity]]

    for frame_num, blobs_in_frame in tqdm(enumerate(blobs_in_video), desc = "detect jumps and crossing"):
        print("frame number ", frame_num)
        for blob_num, blob in enumerate(blobs_in_frame):
            print("blob number ", blob_num)
            if blob.identity == 0:
                # if it is a fish, than it has a portrait and can be assigned
                if blob.is_a_fish:
                    #first we assign the exteme points of the individual fragments
                    if len(blob.next) == 1:
                        blob.identity = blob.next[0].identity
                    elif len(blob.previous) == 1:
                        blob.identity = blob.previous[0].identity
                    else:
                        jumps.append(blob)
                else:
                    crossing_frames.append(frame_num)

    return jumps, crossing_frames

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

    for blobs_in_frame in blobs_in_video:

        for blob in tqdm(blobs_in_frame, desc = "computing velocity model"):
            if blob.is_a_fish_in_a_fragment and current_individual_fragment_identifier != blob.fragment_identifier:
                current_individual_fragment_identifier = blob.fragment_identifier
                distance_travelled_in_individual_fragments.extend(blob.frame_by_frame_velocity())

    return 2 * np.percentile(distance_travelled_in_individual_fragments, percentile)

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
        corresponding_blob_list = [blob for blob in blobs_in_frame if blob.is_a_fish and blob.identity == self.jumping_blob.identity]
        if len(corresponding_blob_list) == 1:
            corresponding_blob = corresponding_blob_list[0]
            velocity = np.linalg.norm(corresponding_blob.centroid - self.jumping_blob.centroid)
            return velocity < self.velocity_threshold
        else:
            self.jumping_blob._identity = None
            return True

    def check_id_availability(self, available_identities, sorted_assignments_indices):
        return [sorted_assignments_index for sorted_assignments_index in sorted_assignments_indices
            if (sorted_assignments_index + 1) in available_identities]

    def check_assigned_identity(self, blobs_in_video, available_identities, sorted_assignments_indices):
        if not self.apply_model_velocity(blobs_in_video):
            print("available_identities ", available_identities)
            print("removing ", self.jumping_blob.identity)
            available_identities.remove(self.jumping_blob.identity)
            print("new_available_identities ", available_identities)
            if len(list(available_identities)) > 0:
                self.jumping_blob.identity = self.check_id_availability(available_identities, sorted_assignments_indices)[0]
                self.check_assigned_identity(blobs_in_video, available_identities, sorted_assignments_indices)

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

def catch_back_crossings(blob):
    try:
        return len(blob.previous[0].next) > 1
    except:
        return False

def get_crossing_blobs(blobs_in_video, crossing_frames):
    """Get the blob objects representing the birth of a crossing at a certain frame
    these crossings has to be disjoint"""
    crossing_blobs = []

    for frame_number in crossing_frames:
        crossing_blobs_in_frame = [blob for blob in blobs_in_video[frame_number]
                                    if len(blob.next) > 1 or catch_back_crossings(blob)]
        if len(crossing_blobs_in_frame) > 0:
            crossing_blobs.append(crossing_blobs_in_frame)

    return crossing_blobs

class Crossing(object):
    def __init__(self, blob = []):
        print("------------------------------------------------------------------")
        self.blob = blob
        self.starting_frame = blob.frame_number
        self.crossing_frames = []
        self.blobs_in_crossing = []

    def get_crossing_frames(self):
        self.crossing_frames.append(self.starting_frame)
        blob = self.blob
        self.blobs_in_crossing.append(blob)
        counter = 1

        #the condition in the while implies that no other blob merged or left the crossing
        while len(blob.next) == 1 and len(blob.next[0].previous) == 1 and not blob.next[0].is_a_fish:
            self.crossing_frames.append(self.starting_frame + counter)
            counter += 1
            blob = blob.next[0]
            self.blobs_in_crossing.append(blob)

        counter = 1

        while len(blob.previous) == 1 and len(blob.previous[0].next) == 1 and not blob.previous[0].is_a_fish:
            self.crossing_frames.append(self.starting_frame - counter)
            counter += 1
            blob = blob.previous[0]
            self.blobs_in_crossing.append(blob)

        print("number of blobs added to crossing ", counter)

    def get_identities_in_crossing(self):
        """After a crossing is born there are few possibilities. On one hand, it
        can be generated by fish that were separated in the previous frame. In
        this case we can get the ids of these fish. On the other hand it can
        consist of the merging of a crossing and a fish, or several crossings.
        """
        blobs_before_crossing = self.find_ids(self.blob, attr = 'previous')
        # print("blobs before crossing ", blobs_before_crossing)
        self.ids_before_crossing = self.get_identities_blobs(blobs_before_crossing)
        print("ids before crossing ", self.ids_before_crossing)
        blobs_after_crossing = self.find_ids(self.blob, attr = 'next')
        # print("blobs after crossing ", blobs_after_crossing)
        self.ids_after_crossing = self.get_identities_blobs(blobs_after_crossing)
        print("ids after crossing ", self.ids_after_crossing)

    def get_crossing_blobs(self):
        return self.blobs_in_crossing

    @staticmethod
    def get_animal_blobs_and_crossing_blobs(list_of_blobs):
        animal_blobs = []
        crossing_blobs = []
        [animal_blobs.append(blob) if blob.identity != 0 and blob.is_a_fish else crossing_blobs.append(blob) for blob in list_of_blobs]
        return animal_blobs, crossing_blobs

    @staticmethod
    def find_ids(crossing_blob, attr = ''):
        blobs_to_check = getattr(crossing_blob, attr)
        # print(blobs_to_check)
        animal_blobs, crossing_blobs = Crossing.get_animal_blobs_and_crossing_blobs(blobs_to_check)
        # print("good blobs ", animal_blobs)
        # print("bad blobs ", crossing_blobs)

        while len(crossing_blobs) > 0:
            for blob in crossing_blobs:
                temp_animals = []
                temp_crossing = []
                temp_animals, temp_crossing = Crossing.get_animal_blobs_and_crossing_blobs(getattr(blob, attr))
                # print("temp animals ", temp_animals)
                # print("temp bad animals ", temp_crossing)
                crossing_blobs = temp_crossing
                if len(temp_animals) > 0:
                    animal_blobs += temp_animals

        # print("final good blobs ", animal_blobs)
        return animal_blobs

    @staticmethod
    def get_identities_blobs(list_of_blobs):
        return [getattr(blob, 'identity') for blob in list_of_blobs]

if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector

    #load video and list of blobs
    video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/conflicto_short/session_1/video_object.npy').item()
    number_of_animals = video.number_of_animals
    list_of_blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/conflicto_short/session_1/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    #get portraits for jumps and frame indices for crossings
    jumps, crossing_frames = get_jumps_and_crossing_frames_arrays(blobs)
    jump_images = [blob.portrait for blob in jumps]
    #assign jumps by restoring the network
    assigner = assign_jumps(jump_images, video)

    ###TODO add velocity constraint!!!
    for i, blob in enumerate(jumps):
        jump = Jump(jumping_blob = blob,
                    number_of_animals = video.number_of_animals,
                    net_prediction = assigner._predictions[i],
                    softmax_probs = assigner._softmax_probs[i])
        jump.assign_jump(blobs)
        blob._identity = jump.jumping_blob.identity
        blob._P1_vector = assigner._softmax_probs[i]
        blob._P2_vector = None


    # get crossing blobs
    crossing_blobs = get_crossing_blobs(blobs, crossing_frames)
    crossings = []

    for blobs_in_crossing in tqdm(crossing_blobs, desc = "generating crossings list"):
        # print("blobs in crossing ", blobs_in_crossing)
        for blob in blobs_in_crossing:
            c = Crossing(blob)
            c.get_crossing_frames()
            c.get_identities_in_crossing()
            blobs_in_crossing = c.get_crossing_blobs()
            for blob in blobs_in_crossing:
                blob._identity = c.ids_before_crossing if len(c.ids_before_crossing) > 1 else c.ids_after_crossing
            crossings.append(c)

    frame_by_frame_identity_inspector(video, blobs)
