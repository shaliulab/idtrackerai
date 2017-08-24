from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
import numpy as np
from tqdm import tqdm
import collections
import logging

from blob import ListOfBlobs
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
from compute_velocity_model import compute_velocity_from_list_of_blobs, compute_model_velocity

VEL_PERCENTILE = 99 #percentile used to model velocity jumps


def get_speed(blobs_in_video):
    number_of_animals = blobs_in_video[0][0].number_of_animals
    number_of_frames = len(blobs_in_video)
    centroid_trajectories = np.ones((number_of_animals,number_of_frames, 2))*np.NaN

    for frame_number, blobs_in_frame in enumerate(tqdm(blobs_in_video, desc = "Computing trajectories")):
        for blob in blobs_in_frame:
            if blob.user_generated_identity is not None:
                blob_identity = blob.user_generated_identity
            elif blob._identity_corrected_solving_duplication is not None:
                blob_identity = blob._identity_corrected_solving_duplication
            else:
                blob_identity = blob.identity

            if (blob_identity is not None) and (blob_identity != 0):
                centroid_trajectories[blob_identity-1, frame_number, :] = blob.centroid

    individual_velocities = np.diff(centroid_trajectories, axis = 1)
    individual_speeds = np.linalg.norm(individual_velocities, axis = 2)
    individual_speeds[np.isnan(individual_speeds)] = -1

    return individual_speeds

def get_frames_with_impossible_speed(blobs_in_video, individual_speeds):
    if not hasattr(video, "velocity_threshold"):
        video.velocity_threshold = compute_model_velocity(blobs, video.number_of_animals, percentile = VEL_PERCENTILE)
    jumps_identities, jumps_frame_numbers = np.where(individual_speeds > video.velocity_threshold)
    velocities = individual_speeds[jumps_identities, jumps_frame_numbers]
    return jumps_identities + 1, jumps_frame_numbers, velocities

class ImpossibleJump(object):
    def __init__(self, blobs_in_video, impossible_jump_identity, impossible_jump_frame, number_of_animals):
        self.blobs = blobs_in_video
        self.blob = [blob for blob in self.blobs[impossible_jump_frame] if blob.identity == impossible_jump_identity]
        self.number_of_animals = number_of_animals

    def compare_fragment_identifiers(self):
        if len(self.blob) == 0: return False
        assert len(self.blob) == 1
        self.blob = self.blob[0]
        if self.blob.is_a_fish_in_a_fragment:
            if not self.blob.previous[0].is_a_crossing and not self.blob.next[0].is_a_crossing:
                return False
        next_fragment_identifiers = [blob_next_frame.fragment_identifier
                                    for blob_next_frame in self.blobs[self.blob.frame_number + 1]
                                    if blob_next_frame.identity == self.blob.identity]
        if len(next_fragment_identifiers) == 1:
            next_fragment_identifier = next_fragment_identifiers[0]
            self.blob_next_frame = blob_next_frame
            return self.blob.fragment_identifier != next_fragment_identifier
        else:
            print("weird next_fragment_identifiers length: ", len(next_fragment_identifiers))
            return False

    def check(self):
        return self.compare_fragment_identifiers() and not self.blob.is_a_ghost_crossing

    @staticmethod
    def get_P2_vectors(blob1, blob2):
        return blob1._P2_vector, blob2._P2_vector

    @staticmethod
    def get_assigned_and_corrected_identity(blob):
        if blob._user_generated_identity is not None:
            return blob._user_generated_identity
        elif blob._identity_corrected_solving_duplication is not None:
            return blob._identity_corrected_solving_duplication
        elif blob.identity is not None:
            return blob.identity
        else:
            return None

    @staticmethod
    def get_identities_assigned_and_corrected_in_frame(blobs_in_frame):
        identities_in_frame = []
        for blob in blobs_in_frame:
            blob_identity = get_assigned_and_corrected_identity(blob)
            if blob_identity is not None:
                identities_in_frame.append(blob_identity)
        return identities_in_frame

    def get_available_identities(self):
        identities_in_frame = set(self.get_identities_assigned_and_corrected_in_frame(self.blobs[blob.frame_number])).remove(0)
        return identities_in_frame, set(range(1, self.number_of_animals + 1)) - identities_in_frame

    def reassign_blob(self, blob):
        non_available_identities, available_identities = self.get_available_identities()
        P2_vector = blob._P2_vector
        P2_vector[list(non_available_identities)] = 0
        candidate_id = np.argmax(P2_vector) + 1
        if candidate_id in available_identities:
            blob.identity = candidate_id
            

    @staticmethod
    def give_me_the_blob_to_correct(list_of_blobs):
        P2_maxima = [np.max(blob._P2_vector) for blob in list_of_blobs]
        minimal_P2s_indices = np.where(P2_maxima == np.min(P2_maxima))[0]
        if len(minimal_P2s_indices) == 1:
            return list_of_blobs[minimal_P2s_indices[0]]
        else:
            raise NotImplementedError("the P2_vecs has same min max")


    def correct_impossible_jumps(self):
        if self.check():
            #evaluate P2 of the two frames
            P2_vector_blob, P2_vector_blob_after_jump = self.get_P2_vectors(self.blob, self.blob_next_frame)
            print("frame number starting jump ", self.blob.frame_number)
            print("blob P2s ", P2_vector_blob)
            print("blob next frame P2 ", P2_vector_blob_after_jump)
            print("________________________________")
            if self.blob.assigned_during_accumulation and self.blob_next_frame.assigned_during_accumulation:
                logging.warn("A velocity jump occured in subsequent frames during accumulation")
            else:
                blob_to_correct = give_me_the_blob_to_correct([self.blob, self.blob_next_frame])



# old check method
# def check(self):
    # if self.compare_fragment_identifiers() and not self.blob.is_a_ghost_crossing:
    #     print("It is an impossible jump")
    #     print("frame number ", self.blob.frame_number)
    #     print("identity ", self.blob.identity)
    #     print("fish_in_a_fragment ", self.blob.is_a_fish_in_a_fragment)
    #     print("jump ", self.blob.is_a_jump)
    #     print("ghost crossing ", self.blob.is_a_ghost_crossing)
    #     print("_______________________________________")
    #     return True
    # return False



if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector
    NUM_CHUNKS_BLOB_SAVING = 10

    #load video and list of blobs
    # video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/video_object.npy').item()
    video = np.load('/home/themis/Desktop/IdTrackerDeep/videos/idTrackerDeep_Sex/mixedGroup_28indiv_first/session_rr05_no_ghost/video_object.npy').item()
    number_of_animals = video.number_of_animals
    # list_of_blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/preprocessing/blobs_collection.npy'
    list_of_blobs_path = '/home/themis/Desktop/IdTrackerDeep/videos/idTrackerDeep_Sex/mixedGroup_28indiv_first/session_rr05_no_ghost/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    individual_speeds = get_speed(blobs)
    jumps_identities, jumps_frame_numbers, velocities = get_frames_with_impossible_speed(blobs, individual_speeds)
    counter = 0

    for identity, frame_number in zip(jumps_identities, jumps_frame_numbers):
        impossible_jump = ImpossibleJump(blobs, identity, frame_number, number_of_animals)
        impossible_jump.correct_impossible_jumps()
